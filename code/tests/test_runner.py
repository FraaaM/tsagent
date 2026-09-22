"""End-to-end episodes on the tiny pool with the scripted stand-in model: outputs, trace
consistency, determinism (FR7), graceful termination and the protocol counter-measures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic_ai import messages as M
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from src.baseline_agent.runner import (
    MESSAGES_FILE,
    SUMMARY_FILE,
    TRACE_FILE,
    VERDICTS_FILE,
    EpisodeConfig,
    compare_runs,
    run_episode,
)
from src.baseline_agent.schemas import (
    ModelRequestRecord,
    RunEndRecord,
    SeriesRecord,
    ToolCallRecord,
    VerdictEvent,
    validate_trace_file,
    validate_verdicts_file,
)


def config(**overrides) -> EpisodeConfig:
    base = dict(model="scripted", total_tokens_limit=None)
    base.update(overrides)
    return EpisodeConfig(**base)


def run(pool, tmp_path: Path, name: str = "run", **overrides):
    return run_episode(pool, config(**overrides), tmp_path / name)


def trace_of(result):
    return validate_trace_file(result.files[TRACE_FILE])


def by_kind(records, cls):
    return [r for r in records if isinstance(r, cls)]


class TestHappyPath:
    def test_outputs_exist_and_validate(self, tiny_pool, tmp_path):
        result = run(tiny_pool, tmp_path)
        for name in (VERDICTS_FILE, TRACE_FILE, MESSAGES_FILE, SUMMARY_FILE):
            assert result.files[name].is_file()
        verdicts = validate_verdicts_file(result.files[VERDICTS_FILE])
        records = trace_of(result)
        end = records[-1]
        assert result.termination.reason == "completed" and end.termination.reason == "completed"
        assert [v.series_id for v in verdicts] == tiny_pool.series_ids  # examination order == pool order
        assert {v.series_id: v.verdict for v in verdicts}[tiny_pool.series_ids[2]] == "anomalous"
        assert end.series.model_dump() == {"in_pool": 5, "examined": 5, "verdicted": 5, "anomalous": 2, "normal": 3, "examined_without_verdict": 0}
        summary = json.loads(result.files[SUMMARY_FILE].read_text(encoding="utf-8"))
        assert summary["run_id"] == result.run_id and summary["series"]["verdicted"] == 5

    def test_trace_is_internally_consistent(self, tiny_pool, tmp_path):
        result = run(tiny_pool, tmp_path)
        records = trace_of(result)
        end: RunEndRecord = records[-1]
        requests = by_kind(records, ModelRequestRecord)
        calls = by_kind(records, ToolCallRecord)
        series = by_kind(records, SeriesRecord)
        messages = json.loads(result.files[MESSAGES_FILE].read_text(encoding="utf-8"))
        n_responses = sum(m["kind"] == "response" for m in messages)
        assert len(requests) == end.requests == n_responses == 7
        assert [r.request_index for r in requests] == list(range(1, 8))
        assert end.tool_calls.model_dump() == {"get_series": 5, "get_pool_meta_data": 1, "failed": 0}
        assert len(calls) == 6 and all(c.ok for c in calls)
        assert [c.request_index for c in calls] == [1, 2, 3, 4, 5, 6]
        assert len(series) == 5 and all(s.get_series_calls == 1 and s.examined and s.verdict for s in series)
        assert all(s.first_request_index is not None and s.first_request_index <= s.last_request_index for s in series)
        # usage: per-request records add up to the totals, and attribution conserves them
        assert sum(r.usage.total_tokens for r in requests) == end.usage.total_tokens > 0
        assert sum(r.usage.input_tokens for r in requests) == end.usage.input_tokens
        attributed = sum(s.attributed_usage.total_tokens for s in series) + end.overhead_usage.total_tokens
        assert attributed == pytest.approx(end.usage.total_tokens)
        assert sum(s.attributed_usage.requests for s in series) + end.overhead_usage.requests == pytest.approx(end.requests)
        assert end.overhead_usage.requests == 2  # the metadata turn and the first (prompt-only) turn
        # every response carries a growing context measure and a verdict per served series
        assert [r.history_chars for r in requests] == sorted(r.history_chars for r in requests)
        assert sum(r.verdicts_accepted for r in requests) == end.verdicts.accepted == 5
        assert by_kind(records, VerdictEvent)[0].request_index == 3

    def test_replay_is_deterministic(self, tiny_pool, tmp_path):
        a = run(tiny_pool, tmp_path, "a")
        b = run(tiny_pool, tmp_path, "b")
        assert a.run_id == b.run_id
        diff = compare_runs(a.output_dir, b.output_dir)
        assert diff.identical, diff.differences
        assert a.files[VERDICTS_FILE].read_bytes() == b.files[VERDICTS_FILE].read_bytes()

    def test_compare_runs_detects_a_difference(self, tiny_pool, tmp_path):
        a = run(tiny_pool, tmp_path, "a")
        b = run(tiny_pool, tmp_path, "b", scripted_options={"max_series": 3})
        assert a.run_id != b.run_id
        diff = compare_runs(a.output_dir, b.output_dir)
        assert not diff.identical and any(VERDICTS_FILE in d for d in diff.differences)

    def test_partial_coverage_is_the_agents_choice(self, tiny_pool, tmp_path):
        result = run(tiny_pool, tmp_path, scripted_options={"max_series": 2})
        end = trace_of(result)[-1]
        assert result.termination.reason == "completed"
        assert end.series.examined == end.series.verdicted == 2 and end.series.in_pool == 5
        assert len(validate_verdicts_file(result.files[VERDICTS_FILE])) == 2

    def test_run_id_reflects_pool_and_config(self, tiny_pool, tmp_path):
        a = run(tiny_pool, tmp_path, "a")
        b = run(tiny_pool, tmp_path, "b", seed=7)
        c = run(tiny_pool.subset(tiny_pool.series_ids[:3]), tmp_path, "c")
        assert len({a.run_id, b.run_id, c.run_id}) == 3

    def test_refuses_to_clobber_an_existing_run(self, tiny_pool, tmp_path):
        run(tiny_pool, tmp_path, "x")
        with pytest.raises(FileExistsError):
            run(tiny_pool, tmp_path, "x")
        run_episode(tiny_pool, config(), tmp_path / "x", overwrite=True)


class TestTermination:
    def test_request_limit_ends_the_episode_gracefully(self, tiny_pool, tmp_path):
        result = run(tiny_pool, tmp_path, request_limit=6, scripted_options={"misbehave": "never_stop"})
        records = trace_of(result)
        end = records[-1]
        assert result.termination.reason == "usage_limit" and "request_limit" in (end.termination.detail or "")
        assert end.requests == 6 == len(by_kind(records, ModelRequestRecord))
        validate_verdicts_file(result.files[VERDICTS_FILE])

    def test_tool_calls_limit(self, tiny_pool, tmp_path):
        result = run(tiny_pool, tmp_path, tool_calls_limit=3, scripted_options={"misbehave": "never_stop"})
        end = trace_of(result)[-1]
        assert end.termination.reason == "usage_limit" and "tool_calls_limit" in (end.termination.detail or "")
        assert end.tool_calls.get_series + end.tool_calls.get_pool_meta_data == 3

    def test_token_limit_after_a_response_still_books_that_response(self, tiny_pool, tmp_path):
        result = run(tiny_pool, tmp_path, total_tokens_limit=3000, scripted_options={"misbehave": "never_stop"})
        records = trace_of(result)
        end = records[-1]
        requests = by_kind(records, ModelRequestRecord)
        messages = json.loads(result.files[MESSAGES_FILE].read_text(encoding="utf-8"))
        assert end.termination.reason == "usage_limit" and "total_tokens_limit" in (end.termination.detail or "")
        # the response that tripped the limit is charged by the provider, so it is booked
        # even when pydantic-ai does not keep it in the history
        assert end.requests == len(requests) == 3 >= sum(m["kind"] == "response" for m in messages)
        assert end.usage.total_tokens > 3000 and requests[-1].usage.total_tokens > 0
        assert sum(r.usage.total_tokens for r in requests) == end.usage.total_tokens

    def test_model_error_is_recorded_and_outputs_survive(self, tiny_pool, tmp_path):
        from src.baseline_agent.scripted import ScriptedPolicy

        inner = ScriptedPolicy()
        calls = {"n": 0}

        def flaky(messages, info):
            calls["n"] += 1
            if calls["n"] == 4:
                raise ModelHTTPError(status_code=400, model_name="flaky", body={"error": "context length exceeded"})
            return inner(messages, info)

        result = run_episode(tiny_pool, config(model="flaky"), tmp_path / "flaky", model=FunctionModel(flaky))
        records = trace_of(result)
        end = records[-1]
        assert end.termination.reason == "model_error" and "context length exceeded" in (end.termination.detail or "")
        assert end.requests == 3
        assert [v.series_id for v in validate_verdicts_file(result.files[VERDICTS_FILE])] == tiny_pool.series_ids[:1]

    def test_exhausted_output_retries_is_a_protocol_error(self, tiny_pool, tmp_path):
        sid = tiny_pool.series_ids[0]

        def stubborn(messages, info):
            last = messages[-1].parts
            if any(isinstance(p, M.UserPromptPart) for p in last):
                return M.ModelResponse(parts=[M.ToolCallPart("get_series", {"series_id": sid})], usage=RequestUsage(input_tokens=1, output_tokens=1))
            return M.ModelResponse(parts=[M.TextPart("DONE 1")], usage=RequestUsage(input_tokens=1, output_tokens=1))

        result = run_episode(tiny_pool, config(model="stubborn", output_retries=1), tmp_path / "s", model=FunctionModel(stubborn))
        records = trace_of(result)
        end = records[-1]
        assert end.termination.reason == "protocol_error" and "retries" in (end.termination.detail or "")
        assert end.retries == 1 and end.series.examined_without_verdict == 1 and end.series.verdicted == 0
        assert [r.is_retry for r in by_kind(records, ModelRequestRecord)] == [False, False, True]


class TestProtocolCounterMeasures:
    def test_missing_verdict_triggers_one_retry(self, tiny_pool, tmp_path):
        result = run(tiny_pool, tmp_path, scripted_options={"misbehave": "skip_last_verdict"})
        records = trace_of(result)
        end = records[-1]
        assert end.termination.reason == "completed" and end.retries == 1
        assert end.series.verdicted == 5 and end.series.examined_without_verdict == 0
        retry_requests = [r for r in by_kind(records, ModelRequestRecord) if r.is_retry]
        assert len(retry_requests) == 1 and retry_requests[0].verdicts_accepted == 1
        messages = json.loads(result.files[MESSAGES_FILE].read_text(encoding="utf-8"))
        retry_parts = [p for m in messages for p in m["parts"] if p["part_kind"] == "retry-prompt"]
        assert len(retry_parts) == 1 and "still have no verdict" in retry_parts[0]["content"]

    def test_enforcement_can_be_switched_off(self, tiny_pool, tmp_path):
        result = run(tiny_pool, tmp_path, enforce_verdicts=False, scripted_options={"misbehave": "skip_last_verdict"})
        end = trace_of(result)[-1]
        assert end.termination.reason == "completed" and end.retries == 0
        assert end.series.examined == 5 and end.series.verdicted == 4 and end.series.examined_without_verdict == 1

    def test_blind_verdict_is_discarded(self, tiny_pool, tmp_path):
        result = run(tiny_pool, tmp_path, scripted_options={"max_series": 3, "misbehave": "blind_verdict"})
        records = trace_of(result)
        end = records[-1]
        rejected = [e for e in by_kind(records, VerdictEvent) if not e.accepted]
        assert len(rejected) == 1 and rejected[0].reason == "not_examined" and not rejected[0].examined
        assert end.verdicts.rejected == 1 and end.series.verdicted == 3
        assert rejected[0].series_id not in {v.series_id for v in validate_verdicts_file(result.files[VERDICTS_FILE])}

    def test_unknown_id_verdict_is_discarded(self, tiny_pool, tmp_path):
        result = run(tiny_pool, tmp_path, scripted_options={"max_series": 2, "misbehave": "unknown_id"})
        records = trace_of(result)
        rejected = [e for e in by_kind(records, VerdictEvent) if not e.accepted]
        assert [e.reason for e in rejected] == ["unknown_series_id"]
        assert records[-1].series.verdicted == 2

    def test_malformed_verdict_line_is_reported_then_recovered(self, tiny_pool, tmp_path):
        result = run(tiny_pool, tmp_path, scripted_options={"misbehave": "bad_json"})
        records = trace_of(result)
        end = records[-1]
        rejected = [e for e in by_kind(records, VerdictEvent) if not e.accepted]
        assert len(rejected) == 1 and rejected[0].reason.startswith("invalid_record")
        assert end.termination.reason == "completed" and end.retries == 1 and end.series.verdicted == 5

    def test_quitting_immediately_is_allowed(self, tiny_pool, tmp_path):
        result = run(tiny_pool, tmp_path, scripted_options={"misbehave": "quit_early"})
        end = trace_of(result)[-1]
        assert end.termination.reason == "completed" and end.retries == 0
        assert end.series.examined == 0 and end.series.verdicted == 0
        assert validate_verdicts_file(result.files[VERDICTS_FILE]) == []

    def test_malformed_series_reaches_the_model_as_text(self, malformed_pool, tmp_path):
        result = run(malformed_pool, tmp_path)
        records = trace_of(result)
        end = records[-1]
        failed = [c for c in by_kind(records, ToolCallRecord) if not c.ok]
        assert [c.series_id for c in failed] == ["M__nan_full", "M__gap_full"]
        assert all("malformed" in (c.error or "") for c in failed)
        series = {s.series_id: s for s in by_kind(records, SeriesRecord)}
        assert series["M__nan_full"].failed_get_series_calls == 1 and not series["M__nan_full"].examined
        assert end.tool_calls.failed == 2 and end.series.examined == 1 and end.series.verdicted == 1
        assert end.termination.reason == "completed"

    def test_invalid_tool_arguments_and_unknown_tools_are_traced(self, tiny_pool, tmp_path):
        sid = tiny_pool.series_ids[0]
        step = {"n": 0}

        def sloppy(messages, info):
            step["n"] += 1
            usage = RequestUsage(input_tokens=1, output_tokens=1)
            if step["n"] == 1:
                return M.ModelResponse(parts=[M.ToolCallPart("get_series", {"series_id": 123})], usage=usage)
            if step["n"] == 2:
                return M.ModelResponse(parts=[M.ToolCallPart("cheap_scan", {})], usage=usage)
            if step["n"] == 3:
                return M.ModelResponse(parts=[M.ToolCallPart("get_series", {"series_id": sid})], usage=usage)
            return M.ModelResponse(parts=[M.TextPart(json.dumps({"series_id": sid, "verdict": "normal"}) + "\nDONE 1")], usage=usage)

        result = run_episode(tiny_pool, config(model="sloppy", tool_retries=2), tmp_path / "sloppy", model=FunctionModel(sloppy))
        records = trace_of(result)
        end = records[-1]
        calls = by_kind(records, ToolCallRecord)
        assert end.termination.reason == "completed" and end.series.verdicted == 1
        assert [(c.tool_name, c.ok) for c in calls] == [("get_series", False), ("cheap_scan", False), ("get_series", True)]
        assert end.tool_calls.failed == 2 and end.tool_calls.get_series == 2
