"""Thread request shapes the LangGraph SDK sends: aliases, supersteps, search and prune."""

from typing import Any

import pytest
from pydantic import ValidationError

from aegra_api.models.threads import (
    Thread,
    ThreadCountRequest,
    ThreadCreate,
    ThreadPruneRequest,
    ThreadSearchRequest,
    ThreadUpdate,
)


class TestThreadCreateAliases:
    """The SDK sends snake_case; older Aegra clients send camelCase. Both must work."""

    def test_accepts_snake_case(self) -> None:
        request = ThreadCreate.model_validate({"thread_id": "t-1", "if_exists": "do_nothing"})

        assert request.thread_id == "t-1"
        assert request.if_exists == "do_nothing"

    def test_accepts_camel_case(self) -> None:
        request = ThreadCreate.model_validate({"threadId": "t-2", "ifExists": "do_nothing"})

        assert request.thread_id == "t-2"
        assert request.if_exists == "do_nothing"

    def test_schema_advertises_the_spec_names(self) -> None:
        """Generated clients read the schema, so it must not advertise the camelCase alias."""
        properties = set(ThreadCreate.model_json_schema()["properties"])

        assert {"thread_id", "if_exists"} <= properties
        assert {"threadId", "ifExists"}.isdisjoint(properties)


class TestSupersteps:
    def test_accepts_values_and_as_node(self) -> None:
        request = ThreadCreate.model_validate({"supersteps": [{"updates": [{"values": {"a": 1}, "as_node": "start"}]}]})

        assert request.supersteps is not None
        assert request.supersteps[0]["updates"][0]["as_node"] == "start"

    def test_rejects_superstep_without_updates(self) -> None:
        with pytest.raises(ValidationError, match="non-empty 'updates' list"):
            ThreadCreate.model_validate({"supersteps": [{}]})

    def test_rejects_empty_updates(self) -> None:
        with pytest.raises(ValidationError, match="non-empty 'updates' list"):
            ThreadCreate.model_validate({"supersteps": [{"updates": []}]})

    def test_rejects_command_rather_than_dropping_it(self) -> None:
        """LangGraph's StateUpdate has no command slot, so accepting it would be a silent no-op."""
        with pytest.raises(ValidationError, match="not supported"):
            ThreadCreate.model_validate({"supersteps": [{"updates": [{"command": {"resume": 1}}]}]})


class TestThreadTTL:
    def test_create_accepts_ttl(self) -> None:
        request = ThreadCreate.model_validate({"ttl": {"ttl": 30, "strategy": "delete"}})

        assert request.ttl is not None
        assert request.ttl.default_ttl == 30

    def test_update_accepts_ttl(self) -> None:
        request = ThreadUpdate.model_validate({"ttl": {"ttl": 15, "strategy": "keep_latest"}})

        assert request.ttl is not None
        assert request.ttl.strategy == "keep_latest"


class TestThreadSearch:
    def test_select_accepts_spec_fields(self) -> None:
        request = ThreadSearchRequest.model_validate({"select": ["thread_id", "status", "values"]})

        assert request.select == ["thread_id", "status", "values"]

    def test_select_rejects_unknown_field(self) -> None:
        with pytest.raises(ValidationError):
            ThreadSearchRequest.model_validate({"select": ["nonsense"]})

    def test_ids_are_accepted(self) -> None:
        request = ThreadSearchRequest.model_validate({"ids": ["t-1", "t-2"]})

        assert request.ids == ["t-1", "t-2"]

    @pytest.mark.parametrize(
        ("field", "value"),
        [("values", {"messages": []}), ("extract", {"alias": "values.x"})],
    )
    def test_state_filters_are_rejected_not_ignored(self, field: str, value: Any) -> None:
        """Aegra keeps state in checkpoints; serving these would need a denormalised column."""
        with pytest.raises(ValidationError, match="not supported"):
            ThreadSearchRequest.model_validate({field: value})

    @pytest.mark.parametrize("empty", [None, {}])
    def test_empty_state_filters_pass_through(self, empty: Any) -> None:
        request = ThreadSearchRequest.model_validate({"values": empty, "extract": empty})

        assert request.values in (None, {})
        assert request.extract in (None, {})


class TestThreadCount:
    def test_accepts_spec_filters(self) -> None:
        request = ThreadCountRequest.model_validate({"status": "idle", "metadata": {"a": 1}})

        assert request.status == "idle"
        assert request.metadata == {"a": 1}

    def test_rejects_unknown_status(self) -> None:
        with pytest.raises(ValidationError):
            ThreadCountRequest.model_validate({"status": "sleeping"})


class TestThreadPrune:
    def test_accepts_thread_ids_and_strategy(self) -> None:
        request = ThreadPruneRequest.model_validate({"thread_ids": ["t-1"], "strategy": "keep_latest"})

        assert request.thread_ids == ["t-1"]
        assert request.strategy == "keep_latest"

    def test_rejects_unknown_strategy(self) -> None:
        with pytest.raises(ValidationError):
            ThreadPruneRequest.model_validate({"strategy": "shred"})

    def test_empty_body_sweeps_expired(self) -> None:
        request = ThreadPruneRequest.model_validate({})

        assert request.thread_ids is None
        assert request.strategy == "delete"


class TestThreadResponseShape:
    def test_declares_the_spec_fields(self) -> None:
        expected = {"values", "interrupts", "config", "ttl", "state_updated_at"}

        assert expected <= set(Thread.model_fields)
