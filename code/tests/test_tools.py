"""Tool registration and behaviour of get_series / get_pool_meta_data (spec 02, FR1-FR3).

No LLM is involved: the agent is driven by pydantic-ai's FunctionModel / TestModel.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic_ai import messages as M
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from src.baseline_agent import TOOL_NAMES, Pool, build_agent, get_pool_meta_data, get_series, read_split_ids, registered_tool_names
from src.baseline_agent.agent import EpisodeDeps
from src.baseline_agent.models import build_model
from src.baseline_agent.prompt import USER_PROMPT
from src.baseline_agent.tools import is_error
from src.baseline_agent.trace import TraceRecorder
from src.baseline_agent.verdicts import VerdictCollector
from src.baseline_agent.writers import JsonlWriter


def make_deps(pool: Pool, tmp_path: Path) -> tuple[EpisodeDeps, JsonlWriter]:
    writer = JsonlWriter(tmp_path / "trace.jsonl")
    deps = EpisodeDeps(pool=pool, recorder=TraceRecorder(writer, pool), collector=VerdictCollector())
    return deps, writer


# --- FR3: exactly two tools -------------------------------------------------------------


class TestRegistration:
    def test_exactly_the_two_spec_tools_are_registered(self):
        agent = build_agent(build_model("test"))
        assert sorted(registered_tool_names(agent)) == sorted(TOOL_NAMES)
        assert set(TOOL_NAMES) == {"get_series", "get_pool_meta_data"}

    def test_model_sees_two_function_tools_and_no_output_tool(self, tiny_pool, tmp_path):
        seen: list[AgentInfo] = []

        def policy(messages, info: AgentInfo) -> M.ModelResponse:
            seen.append(info)
            return M.ModelResponse(parts=[M.TextPart("DONE 0")], usage=RequestUsage(input_tokens=1, output_tokens=1))

        agent = build_agent(FunctionModel(policy))
        deps, writer = make_deps(tiny_pool, tmp_path)
        agent.run_sync(USER_PROMPT, deps=deps)
        writer.close()
        assert len(seen) == 1
        assert sorted(t.name for t in seen[0].function_tools) == sorted(TOOL_NAMES)
        assert seen[0].output_tools == []
        assert seen[0].allow_text_output is True

    def test_tool_descriptions_do_not_promise_analysis(self, tiny_pool, tmp_path):
        seen: list[AgentInfo] = []

        def policy(messages, info: AgentInfo) -> M.ModelResponse:
            seen.append(info)
            return M.ModelResponse(parts=[M.TextPart("DONE 0")], usage=RequestUsage(input_tokens=1, output_tokens=1))

        agent = build_agent(FunctionModel(policy))
        deps, writer = make_deps(tiny_pool, tmp_path)
        agent.run_sync(USER_PROMPT, deps=deps)
        writer.close()
        by_name = {t.name: t for t in seen[0].function_tools}
        assert list(by_name["get_series"].parameters_json_schema["properties"]) == ["series_id"]
        assert by_name["get_series"].parameters_json_schema["properties"]["series_id"]["type"] == "string"
        assert by_name["get_pool_meta_data"].parameters_json_schema.get("properties", {}) == {}
        for tool in by_name.values():
            assert "anomal" not in (tool.description or "").lower().replace("anomaly", "")  # no scoring/detection promised
            assert "score" not in (tool.description or "").lower()


# --- FR1: get_series ------------------------------------------------------------------


class TestGetSeries:
    def test_returns_the_full_raw_series(self, tiny_pool):
        sid = tiny_pool.series_ids[2]
        series = tiny_pool.get(sid)
        text = get_series(tiny_pool, sid)
        assert not is_error(text)
        doc = json.loads(text)
        assert set(doc) == {"series_id", "length", "time_index", "value"}
        assert doc["series_id"] == sid
        assert doc["length"] == len(series) == len(doc["value"]) == len(doc["time_index"])
        assert doc["time_index"] == list(range(len(series)))
        assert doc["value"] == series.value.tolist()
        assert doc["value"][120] == 25.0  # the spike is delivered raw, not smoothed away

    def test_no_label_or_metadata_leaks(self, tiny_parquet):
        pool = Pool.from_parquet(tiny_parquet)
        assert pool.name == "T1"
        for sid in pool.series_ids:
            text = get_series(pool, sid)
            assert "label" not in text
            assert set(json.loads(text)) == {"series_id", "length", "time_index", "value"}
        assert "label" not in get_pool_meta_data(pool)
        assert not hasattr(pool.get(pool.series_ids[0]), "label")

    def test_unknown_id_gives_a_textual_error_not_an_exception(self, tiny_pool):
        text = get_series(tiny_pool, "T1__SYNTH__does-not-exist_full")
        assert is_error(text)
        assert "unknown series_id" in text and "get_pool_meta_data" in text
        with pytest.raises(json.JSONDecodeError):
            json.loads(text)

    @pytest.mark.parametrize("bad", ["", "   "])
    def test_blank_id_is_an_error(self, tiny_pool, bad):
        assert is_error(get_series(tiny_pool, bad))

    def test_malformed_series_give_textual_errors(self, malformed_pool):
        assert not is_error(get_series(malformed_pool, "M__a_full"))
        nan = get_series(malformed_pool, "M__nan_full")
        assert is_error(nan) and "non-finite" in nan and "malformed" in nan
        gap = get_series(malformed_pool, "M__gap_full")
        assert is_error(gap) and "not contiguous" in gap

    def test_wire_format_is_compact_and_deterministic(self, tiny_pool):
        sid = tiny_pool.series_ids[0]
        a, b = get_series(tiny_pool, sid), get_series(tiny_pool, sid)
        assert a == b
        assert ": " not in a and ", " not in a
        assert a.startswith('{"series_id":"')

    def test_value_precision_is_opt_in(self, tiny_pool):
        sid = tiny_pool.series_ids[0]
        raw = json.loads(get_series(tiny_pool, sid))["value"]
        rounded = json.loads(get_series(tiny_pool, sid, value_precision=2))["value"]
        assert rounded == [round(v, 2) for v in raw]
        assert rounded != raw


# --- FR2: get_pool_meta_data ------------------------------------------------------------


class TestGetPoolMetaData:
    def test_lists_every_id_in_pool_order(self, tiny_pool):
        doc = json.loads(get_pool_meta_data(tiny_pool))
        assert set(doc) == {"pool", "n_series", "series_ids"}
        assert doc["pool"] == "T1"
        assert doc["n_series"] == len(tiny_pool) == 5
        assert doc["series_ids"] == tiny_pool.series_ids
        assert len(set(doc["series_ids"])) == len(doc["series_ids"])

    def test_split_selection_reads_only_ids_and_split(self, tiny_pool, tmp_path):
        meta = pd.DataFrame(
            {
                "series_id": tiny_pool.series_ids,
                "y_i": [0, 0, 1, 0, 1],
                "split": ["dev", "test", "dev", "test", "test"],
            }
        )
        path = tmp_path / "T1_metadata.parquet"
        meta.to_parquet(path, index=False)
        dev = read_split_ids(path, "dev")
        assert dev == [tiny_pool.series_ids[0], tiny_pool.series_ids[2]]
        sub = tiny_pool.subset(dev)
        assert sub.series_ids == dev
        with pytest.raises(ValueError):
            read_split_ids(path, "validation")


# --- the agent-facing wrappers ----------------------------------------------------------


class TestAgentWrappers:
    def test_calls_are_timed_traced_and_mark_examination(self, tiny_pool, tmp_path):
        sid = tiny_pool.series_ids[1]

        def policy(messages, info):
            last = messages[-1].parts
            if any(isinstance(p, M.UserPromptPart) for p in last):
                return M.ModelResponse(parts=[M.ToolCallPart("get_pool_meta_data", {})], usage=RequestUsage(input_tokens=1, output_tokens=1))
            if any(isinstance(p, M.ToolReturnPart) and p.tool_name == "get_pool_meta_data" for p in last):
                return M.ModelResponse(parts=[M.ToolCallPart("get_series", {"series_id": sid}), M.ToolCallPart("get_series", {"series_id": "nope"})], usage=RequestUsage(input_tokens=1, output_tokens=1))
            return M.ModelResponse(parts=[M.TextPart(json.dumps({"series_id": sid, "verdict": "normal"}) + "\nDONE 1")], usage=RequestUsage(input_tokens=1, output_tokens=1))

        agent = build_agent(FunctionModel(policy))
        deps, writer = make_deps(tiny_pool, tmp_path)
        result = agent.run_sync(USER_PROMPT, deps=deps)
        writer.close()
        assert "DONE 1" in result.output
        assert dict(deps.collector.examined) == {sid: 1}
        calls = deps.recorder.tool_calls
        assert [c.tool_name for c in calls] == ["get_pool_meta_data", "get_series", "get_series"]
        ok, bad = calls[1], calls[2]
        assert ok.ok and ok.series_id == sid and ok.n_points == len(tiny_pool.get(sid)) and ok.error is None
        assert ok.payload_chars == len(get_series(tiny_pool, sid))
        assert ok.duration_ms >= 0 and ok.request_index == 2 and ok.tool_call_id
        assert not bad.ok and bad.series_id == "nope" and "unknown series_id" in (bad.error or "")
        lines = [json.loads(l) for l in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [l["kind"] for l in lines] == ["tool_call"] * 3

    def test_pool_from_parquet_rejects_missing_columns(self, tmp_path):
        frame = pd.DataFrame({"series_id": ["a"] * 3, "value": np.arange(3.0)})
        path = tmp_path / "broken.parquet"
        frame.to_parquet(path, index=False)
        with pytest.raises(ValueError, match="time_index"):
            Pool.from_parquet(path)

    def test_pool_fingerprint_tracks_content(self, tiny_pool):
        frame = pd.DataFrame(
            {"series_id": ["x"] * 3 + ["y"] * 2, "time_index": [0, 1, 2, 0, 1], "value": [1.0, 2.0, 3.0, 4.0, 5.0]}
        )
        a = Pool.from_frame(frame, "P")
        b = Pool.from_frame(frame.copy(), "P")
        frame.loc[0, "value"] = 1.5
        c = Pool.from_frame(frame, "P")
        assert a.fingerprint == b.fingerprint != c.fingerprint
        assert a.series_ids == ["x", "y"] and a.subset(["y"]).series_ids == ["y"]
        with pytest.raises(ValueError, match="duplicate"):
            Pool("D", [a.get("x"), a.get("x")])
