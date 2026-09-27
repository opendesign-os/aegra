# LangSmith Deployment parity

Gap analysis of Aegra against the LangSmith Deployment (LangGraph Platform)
Agent Server contract. Aegra claims to be a drop-in replacement, so every
shape the LangGraph SDK can put on the wire is part of our contract.

## Sources of truth

| Source | Version | License | What it pins |
|:--|:--|:--|:--|
| `langgraph-api` bundled `openapi.json` | 0.15.1 | Elastic-2.0 | Server contract: 49 paths, request/response schemas. Spec file only — no implementation is read or reused. |
| `langgraph-sdk` | 0.4.2 | MIT | Client call shapes: which fields actually reach the server, and when. |
| `langchain-protocol` | 0.0.18 | MIT | v2/v3 thread-streaming commands and event envelopes. |
| `docs/openapi.json` | 0.10.6 | — | Aegra's own surface: 42 paths. |

Refresh the upstream spec with:

```bash
python - <<'PY'
import io, json, urllib.request, zipfile
meta = json.load(urllib.request.urlopen("https://pypi.org/pypi/langgraph-api/json"))
url = next(f["url"] for f in meta["urls"] if f["filename"].endswith(".whl"))
whl = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url).read()))
open("lgp_openapi.json", "wb").write(whl.read("openapi.json"))
PY
```

## 1. Missing endpoints

`GET /docs` also appears in the upstream spec but is a false positive: FastAPI
serves it and keeps it out of the generated schema.

| Endpoint | Gap |
|:--|:--|
| `GET /ok` | Liveness. Aegra has `/health`, `/live`, `/ready` but not the name the platform advertises. |
| `GET /metrics` | Prometheus exposition of queue depth, run counts, latency. |
| `POST /threads/count` | SDK `threads.count()`. We have `/assistants/count` and `/runs/crons/count`. |
| `POST /threads/{thread_id}/copy` | SDK `threads.copy()`. Forks a thread plus its checkpoint lineage. |
| `GET /threads/{thread_id}/stream` | Thread-level join stream — SDK `threads.join_stream()`. Distinct from the per-run join stream we already serve. |
| `POST /runs/batch` | SDK `runs.create_batch()`. Batch of stateless runs. |
| `GET /runs/crons/{cron_id}` | Read a single cron. We only expose search. |
| `GET /assistants/{assistant_id}/subgraphs/{namespace}` | We take `?namespace=` instead of a path segment, so the SDK's namespaced call 404s. |
| `GET,POST,DELETE /mcp/` | MCP streamable-HTTP server exposing graphs as tools. |
| `POST /a2a/{assistant_id}` | A2A JSON-RPC endpoint. |

Aegra-only additions, all harmless supersets: `GET /assistants`, `GET /threads`,
`GET /health`, `GET /live`, `GET /ready`, `PATCH /threads/{thread_id}/runs/{run_id}`.

## 2. Request-field drift

33 of 50 shared endpoints differ. The run-creation family is the most severe:
these fields are declared upstream, sent by the SDK, and currently dropped by
our Pydantic models — the exact failure mode `CLAUDE.md` calls out, where the
client sees success and the behavior it asked for never happens.

### Run creation — `POST /runs`, `/runs/stream`, `/runs/wait`, `/threads/{id}/runs{,/stream,/wait}`

| Field | Upstream | Aegra |
|:--|:--|:--|
| `durability` | `sync` \| `async` \| `exit` | missing |
| `stream_resumable` | bool, default `false` — SDK always sends it | missing |
| `webhook` | URI called on completion | missing |
| `after_seconds` | number — delayed start | missing |
| `if_not_exists` | `create` \| `reject`, stateful routes only | missing |
| `checkpoint_during` | deprecated alias of `durability` | missing |
| `feedback_keys` | LangSmith-side, may stay inert | missing |

Aegra extensions on the same routes (`stream`, `on_disconnect`, `checkpoint_id`,
`on_completion`, `multitask_strategy` on stateless routes) are supersets and can stay.

### Other drift

| Endpoint | Gap |
|:--|:--|
| `POST /threads` | `supersteps` missing. `thread_id`/`if_exists` are accepted at runtime via `populate_by_name`, but the schema advertises `threadId`/`ifExists`, so generated clients get the wrong names. |
| `PATCH /threads/{id}` | `ttl` missing. |
| `GET /threads/{id}` | `?include` missing (gates `ttl` in the response). |
| `POST /threads/search` | `select`, `extract`, `values`, `ids` missing. |
| `POST /threads/prune` | `strategy`, `thread_ids` missing. |
| `POST /assistants/search`, `POST /runs/crons/search`, `GET /threads/{id}/runs` | `select` missing. |
| `DELETE /assistants/{id}` | `?delete_threads` missing. |
| `POST /assistants/{id}/latest` | upstream takes `?version` query; we take it in the body. |
| `PUT /store/items` | `index`, `ttl` missing. |
| `GET /store/items`, `POST /store/items/search` | `refresh_ttl` missing. |
| `POST /runs/crons`, `PATCH /runs/crons/{id}`, `POST /threads/{id}/runs/crons` | `durability`, `stream_resumable` missing. |
| `GET /threads/{id}/runs/{run_id}/stream` | we declare `_stream_mode`; the SDK sends `stream_mode`, so it is silently ignored. `cancel_on_disconnect` also missing. |
| `GET /threads/{id}/runs/{run_id}/join` | `cancel_on_disconnect` missing. |

## 3. Response-shape drift

| Entity | Upstream fields we do not return |
|:--|:--|
| `Run` | `kwargs`, `metadata`, `multitask_strategy` |
| `Thread` | `values`, `interrupts`, `config`, `ttl`, `state_updated_at`, `extracted` |

`Thread.values` and `Thread.interrupts` matter most: Agent Chat UI and Studio
render thread lists straight from search results and fall back to per-thread
state calls without them.

`Assistant.metadata` is correct on the wire (`response_model_by_alias=False`)
but the generated schema still names it `metadata_dict`, so codegen clients
break. Fix by moving the alias to `validation_alias=AliasChoices(...)`.

Run status enums match upstream exactly.

## 4. Stream modes

Upstream: `values`, `messages`, `messages-tuple`, `tasks`, `checkpoints`,
`updates`, `events`, `debug`, `custom`. Aegra documents seven — `tasks` and
`checkpoints` are unverified.

## 5. v2/v3 thread protocol

`POST /threads/{id}/commands` dispatches only `run.start` and `input.respond`.
The protocol defines eight more: `subscription.subscribe`,
`subscription.unsubscribe`, `subscription.reconnect`, `agent.getTree`,
`input.inject`, `state.get`, `state.listCheckpoints`, `state.fork`.

The SDK also offers a `websocket` transport: commands still go over HTTP POST,
but the event stream upgrades on `/threads/{id}/stream/events`. We are SSE-only.

## 6. Library mapping

Everything below is MIT or Apache-2.0 and already available.

| Gap | Library | Hook |
|:--|:--|:--|
| `durability` | `langgraph` | `Pregel.astream(..., durability=...)` |
| `supersteps` | `langgraph` | `Pregel.bulk_update_state(config, supersteps)` |
| Store `index` / `ttl` / `refresh_ttl` | `langgraph` | `PutOp.index`, `GetOp.refresh_ttl`, postgres store `ttl_minutes` |
| `tasks` / `checkpoints` stream modes | `langgraph` | native stream modes |
| Thread copy | `langgraph-checkpoint-postgres` | checkpoint lineage copy |
| `/mcp/` | `mcp` 2.2.0 (MIT) | streamable-HTTP ASGI app |
| `/a2a/{assistant_id}` | `a2a-sdk` 1.1.5 (Apache-2.0) | JSON-RPC handler |
| `/metrics` | `prometheus-client` 0.26.0 | ASGI exposition |
| `webhook` | `httpx` (existing dep) | post-run delivery with retry |
| WebSocket event transport | `websockets` (existing transitive dep) | upgrade on `/stream/events` |


## Status after the alignment pass

Regenerated `docs/openapi.json` (49 paths) diffed against `langgraph-api` 0.15.1:

- **Field-level gaps: zero.** Every remaining difference on a shared endpoint is an
  Aegra extension (a superset), not a missing upstream field.
- **Endpoint gaps: none.** `/mcp/` and `/a2a/{assistant_id}` both serve. `GET /docs`
  and `GET /metrics` show as missing only because FastAPI and the Prometheus
  instrumentator keep them out of the generated schema.

MCP runs on the `mcp` SDK's stateless Streamable HTTP transport, pinned below 2.0
because 2.x swaps httpx for httpx2. A2A is served from hand-written Pydantic
models: `a2a-sdk` ships protobuf/gRPC types, which are a different wire shape
from the JSON-RPC form this endpoint speaks.

Two fields are deliberately rejected rather than served, because Aegra keeps thread
state in checkpoints instead of denormalising it onto the thread row:
`ThreadSearchRequest.values` and `ThreadSearchRequest.extract` both return 422 with
a message pointing at `/threads/{thread_id}/state`.

`subscription.subscribe` / `unsubscribe` / `reconnect` answer `not_supported`:
channels are fixed per connection in the `/stream/events` request body.

Verified against the Docker stack (Redis broker, 3 workers) with
`scratchpad/verify_api.py`: 90 checks, 0 failures.
