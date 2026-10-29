"""Exercise every Agent Protocol endpoint against a running Aegra server."""

from __future__ import annotations

import json
import sys
import uuid
from typing import Any

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:2026"
OUT = sys.argv[2] if len(sys.argv) > 2 else "verify_results.json"

client = httpx.Client(base_url=BASE, timeout=60.0)
results: list[tuple[str, str, Any, str]] = []
state: dict[str, Any] = {}


def call(label: str, method: str, path: str, expect: tuple[int, ...] = (200, 201), **kw: Any) -> httpx.Response | None:
    try:
        r = client.request(method, path, **kw)
    except Exception as e:
        results.append((label, method + " " + path, "EXC", type(e).__name__ + ": " + str(e)[:120]))
        return None
    ok = r.status_code in expect
    detail = "OK" if ok else r.text[:160].replace("\n", " ")
    results.append((label, method + " " + path, r.status_code, detail))
    return r


def field(r: httpx.Response | None, key: str) -> Any:
    if r is None or r.status_code >= 300:
        return None
    try:
        data = r.json()
    except Exception:
        return None
    if isinstance(data, list):
        data = data[0] if data else {}
    return data.get(key) if isinstance(data, dict) else None


# --- system ---------------------------------------------------------------
call("system", "GET", "/ok")
call("system", "GET", "/info")
call("system", "GET", "/health")
call("system", "GET", "/live")
call("system", "GET", "/ready")
call("system", "GET", "/metrics", (200, 404))
call("system", "GET", "/openapi.json")

# --- assistants -----------------------------------------------------------
aid = "asst-" + uuid.uuid4().hex[:8]
r = call(
    "assistants",
    "POST",
    "/assistants",
    json={
        "assistant_id": aid,
        "graph_id": "agent",
        "name": "verify",
        "config": {"configurable": {"probe_id": aid}},
        "metadata": {"probe": "1"},
    },
)
state["assistant_id"] = field(r, "assistant_id") or aid
A = state["assistant_id"]
call("assistants", "GET", "/assistants")
call("assistants", "POST", "/assistants/search", json={"limit": 5})
call("assistants", "POST", "/assistants/count", json={})
r = call("assistants", "GET", "/assistants/" + A)
if r is not None and r.status_code < 300:
    body = r.json()
    has = "metadata" in body and "metadata_dict" not in body
    results.append(
        (
            "assistants",
            "Assistant.metadata wire name",
            200 if has else "MISS",
            "OK" if has else "serialized as " + ",".join(body.keys()),
        )
    )
call("assistants", "PATCH", "/assistants/" + A, json={"name": "verify2"})
call("assistants", "GET", "/assistants/" + A + "/schemas")
call("assistants", "GET", "/assistants/" + A + "/graph")
call("assistants", "GET", "/assistants/" + A + "/subgraphs")
call("assistants", "GET", "/assistants/" + A + "/subgraphs/inner", (200, 404))
call("assistants", "POST", "/assistants/" + A + "/versions", json={})
call("assistants", "POST", "/assistants/" + A + "/latest", json={"version": 1})

# --- threads --------------------------------------------------------------
r = call(
    "threads",
    "POST",
    "/threads",
    json={"metadata": {"probe": "1", "graph_id": "agent", "assistant_id": A}, "if_exists": "raise"},
)
T = field(r, "thread_id")
state["thread_id"] = T
call("threads", "GET", "/threads")
call("threads", "POST", "/threads/search", json={"limit": 5})
call("threads", "POST", "/threads/count", json={})
call("threads", "GET", "/threads/" + str(T))
call("threads", "PATCH", "/threads/" + str(T), json={"metadata": {"probe": "2"}})
call("threads", "GET", "/threads/" + str(T) + "/state")
call("threads", "POST", "/threads/" + str(T) + "/state", json={"values": {"messages": []}})
call("threads", "POST", "/threads/" + str(T) + "/history", json={"limit": 3})
call("threads", "GET", "/threads/" + str(T) + "/history")
r = call("threads", "POST", "/threads/" + str(T) + "/copy")
state["copy_id"] = field(r, "thread_id")
call(
    "threads",
    "POST",
    "/threads",
    json={
        "thread_id": str(uuid.uuid4()),
        "metadata": {"graph_id": "agent"},
        "supersteps": [{"updates": [{"values": {"messages": []}, "as_node": "__start__"}]}],
    },
    expect=(200, 201),
)

# --- runs -----------------------------------------------------------------
run_body = {
    "assistant_id": A,
    "input": {"messages": [{"role": "user", "content": "ping"}]},
    "stream_resumable": False,
    "durability": "async",
    "after_seconds": 0,
    "if_not_exists": "create",
}
r = call("runs", "POST", "/threads/" + str(T) + "/runs", json=run_body)
R = field(r, "run_id")
if r is not None and r.status_code < 300:
    body = r.json()
    for key in ("kwargs", "metadata", "multitask_strategy"):
        present = key in body
        results.append(
            ("runs", "Run." + key + " in response", 200 if present else "MISS", "OK" if present else "absent")
        )
call("runs", "GET", "/threads/" + str(T) + "/runs")
call("runs", "GET", "/threads/" + str(T) + "/runs/" + str(R))
call("runs", "POST", "/threads/" + str(T) + "/runs/" + str(R) + "/cancel", (200, 204, 404, 409))
call("runs", "POST", "/runs/cancel", (200, 204), json={"status": "pending"})
call("runs", "POST", "/runs/batch", (200, 201, 404, 405), json=[{"assistant_id": A, "input": {"messages": []}}])
call("runs", "DELETE", "/threads/" + str(T) + "/runs/" + str(R), (200, 204, 404, 409))

# --- contract rejections ---------------------------------------------------
call(
    "contract",
    "POST",
    "/threads/" + str(T) + "/runs",
    (422,),
    json=dict(run_body, checkpoint_during=True, durability="sync"),
)
call("contract", "POST", "/threads/" + str(T) + "/runs", (422,), json=dict(run_body, webhook="ftp://bad/x"))
call("contract", "POST", "/threads/" + str(T) + "/runs", (422,), json=dict(run_body, durability="never"))

# --- crons ----------------------------------------------------------------
r = call(
    "crons",
    "POST",
    "/runs/crons",
    (200, 201),
    json={
        "assistant_id": A,
        "schedule": "0 0 * * *",
        "input": {"messages": []},
        "enabled": False,
    },
)
C = field(r, "cron_id")
call("crons", "POST", "/runs/crons/search", json={"limit": 5})
call("crons", "POST", "/runs/crons/count", json={})
if C:
    call("crons", "GET", "/runs/crons/" + str(C))
    call("crons", "PATCH", "/runs/crons/" + str(C), json={"enabled": False})
    call("crons", "DELETE", "/runs/crons/" + str(C), (200, 204))

# --- store ----------------------------------------------------------------
ns = ["verify", uuid.uuid4().hex[:6]]
call("store", "PUT", "/store/items", (200, 201, 204), json={"namespace": ns, "key": "k1", "value": {"v": 1}})
call("store", "GET", "/store/items", params={"namespace": ".".join(ns), "key": "k1"})
call("store", "POST", "/store/items/search", json={"namespace_prefix": ns, "limit": 5})
call("store", "POST", "/store/namespaces", json={"prefix": ["verify"], "limit": 5})
call("store", "DELETE", "/store/items", (200, 204), json={"namespace": ns, "key": "k1"})

# --- v2 / v3 protocol ------------------------------------------------------
for cid, method, want in (
    (1, "agent.getTree", "success"),
    (2, "state.get", "success"),
    (3, "state.listCheckpoints", "success"),
    (4, "subscription.subscribe", "not_supported"),
):
    rr = call(
        "v2", "POST", "/threads/" + str(T) + "/commands", (200, 503), json={"id": cid, "method": method, "params": {}}
    )
    if rr is not None and rr.status_code == 200:
        payload = rr.json()
        kind = payload.get("type") if isinstance(payload, dict) else None
        err = payload.get("error") if isinstance(payload, dict) else None
        code = err.get("code") if isinstance(err, dict) else err
        got = kind if kind == "success" else str(code)
        results.append(
            ("v2", method + " -> " + want, 200 if got == want else "MISS", "OK" if got == want else "got " + got)
        )
call("v2", "GET", "/threads/" + str(T) + "/stream", (200, 404, 405))

# --- newly aligned surface -------------------------------------------------
call("aligned", "GET", "/threads/" + str(T) + "/stream", (200, 404))
call("aligned", "GET", "/threads/" + str(T), params={"include": ["ttl", "values", "interrupts"]})
call("aligned", "GET", "/threads/" + str(T), (422,), params={"include": ["bogus"]})
call("aligned", "POST", "/threads/search", json={"limit": 5, "select": ["thread_id", "status"]})
call("aligned", "POST", "/threads/search", json={"limit": 5, "ids": [str(T)]})
call("aligned", "POST", "/threads/search", (422,), json={"values": {"a": 1}})
call("aligned", "POST", "/threads/search", (422,), json={"extract": {"a": "values.x"}})
call("aligned", "POST", "/assistants/search", json={"limit": 5, "select": ["assistant_id", "name"]})
call("aligned", "POST", "/assistants/" + A + "/latest", params={"version": 1})
call(
    "aligned",
    "PUT",
    "/store/items",
    (200, 201, 204),
    json={"namespace": ns, "key": "k2", "value": {"v": 2}, "index": ["v"], "ttl": 60},
)
call("aligned", "GET", "/store/items", params={"namespace": ".".join(ns), "key": "k2", "refresh_ttl": True})
call("aligned", "POST", "/store/items/search", json={"namespace_prefix": ns, "limit": 5, "refresh_ttl": True})
call("aligned", "DELETE", "/store/items", (200, 204), json={"namespace": ns, "key": "k2"})
call(
    "aligned",
    "POST",
    "/runs/crons",
    (200, 201),
    json={
        "assistant_id": A,
        "schedule": "0 0 * * *",
        "input": {"messages": []},
        "enabled": False,
        "durability": "exit",
        "stream_resumable": True,
    },
)
for cid, method in ((10, "state.fork"), (11, "input.inject"), (12, "subscription.subscribe")):
    rr = call(
        "aligned",
        "POST",
        "/threads/" + str(T) + "/commands",
        (200, 503),
        json={"id": cid, "method": method, "params": {"checkpoint_id": "x", "message": {"content": "hi"}}},
    )
    if rr is not None and rr.status_code == 200:
        payload = rr.json()
        err = payload.get("error") if isinstance(payload, dict) else None
        code = err.get("code") if isinstance(err, dict) else err
        known = payload.get("type") == "success" or code != "unknown_command"
        results.append(
            ("aligned", method + " recognised", 200 if known else "MISS", "OK" if known else "still unknown_command")
        )

r = call("aligned", "POST", "/threads", json={"metadata": {"probe": "prune"}})
P = field(r, "thread_id")
if P:
    call("aligned", "POST", "/threads/prune", json={"thread_ids": [str(P)], "strategy": "delete"})

# --- cleanup --------------------------------------------------------------
for t in (state.get("copy_id"), T):
    if t:
        call("cleanup", "DELETE", "/threads/" + str(t), (200, 204, 404))
call("cleanup", "DELETE", "/assistants/" + A, (200, 204, 404), params={"delete_threads": True})

# --- report ---------------------------------------------------------------
width = max(len(x[1]) for x in results) + 2
passed = failed = 0
current = None
for group, route, code, detail in results:
    if group != current:
        print("\n--- " + group + " ---")
        current = group
    ok = detail == "OK"
    if ok:
        passed += 1
    else:
        failed += 1
    print("  [" + ("PASS" if ok else "FAIL") + "] " + route.ljust(width) + str(code) + "  " + ("" if ok else detail))

print("\n" + str(passed) + " passed, " + str(failed) + " failed, " + str(len(results)) + " total")
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump([{"group": g, "route": r, "code": c, "detail": d} for g, r, c, d in results], fh, indent=1)
sys.exit(1 if failed else 0)
