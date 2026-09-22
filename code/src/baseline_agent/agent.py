"""The pydantic-ai agent of spec 02: two tools, greedy decoding, a request cap, one check.

What the harness adds on top of the two pure tool functions in ``tools.py`` is
bookkeeping only: timing and a trace record around every call (FR6), the record of which
series were examined, and an output validator that refuses to end the episode while a
retrieved series still has no verdict (FR4/FR5: "a verdict for each series it examines").
The validator never asks for more coverage of the pool - stopping early is the agent's own
decision and is left alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, ModelRetry, RunContext, Tool
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from .pool import Pool
from .prompt import SYSTEM_PROMPT
from .tools import get_pool_meta_data as _pool_meta_data
from .tools import get_series as _get_series
from .trace import TraceRecorder
from .verdicts import ParsedVerdict, VerdictCollector

TOOL_NAMES: tuple[str, ...] = ("get_series", "get_pool_meta_data")
AGENT_NAME = "raw-retrieval-baseline"


@dataclass
class EpisodeDeps:
    """Everything a tool call or the output validator may touch during one episode."""

    pool: Pool
    recorder: TraceRecorder
    collector: VerdictCollector
    value_precision: int | None = None
    enforce_verdicts: bool = True


def get_series(ctx: RunContext[EpisodeDeps], series_id: str) -> str:
    """Return the full raw series with the given id.

    The result is a JSON document with the fields series_id, length, time_index (integer
    positions starting at 0) and value (one number per position). An unknown or malformed
    series_id yields an error message instead of data.

    Args:
        series_id: Identifier of the series, exactly as listed by get_pool_meta_data.
    """
    deps = ctx.deps
    with deps.recorder.tool_call(
        "get_series", ctx.tool_call_id, {"series_id": series_id}, request_index=ctx.run_step, series_id=series_id
    ) as call:
        text = _get_series(deps.pool, series_id, value_precision=deps.value_precision)
        call.finish(text, n_points=deps.pool.length_of(series_id) if series_id in deps.pool else None)
    if call.ok:
        deps.collector.mark_examined(series_id)
    return text


def get_pool_meta_data(ctx: RunContext[EpisodeDeps]) -> str:
    """Return the pool's metadata: its name, its size and the list of all series ids."""
    deps = ctx.deps
    with deps.recorder.tool_call("get_pool_meta_data", ctx.tool_call_id, {}, request_index=ctx.run_step) as call:
        text = _pool_meta_data(deps.pool)
        call.finish(text)
    return text


def build_agent(
    model: Model | str,
    *,
    model_settings: ModelSettings | None = None,
    instructions: str = SYSTEM_PROMPT,
    tool_retries: int = 1,
    output_retries: int = 2,
    name: str = AGENT_NAME,
) -> Agent[EpisodeDeps, str]:
    """One agent, exactly two function tools, plain-text output.

    ``model_settings`` is where greedy decoding lives (``temperature=0``, ``seed``);
    the request cap is a run-time ``UsageLimits`` (see ``runner.py``), not an agent
    attribute. ``output_type`` stays ``str`` on purpose: an output tool would be a third
    tool exposed to the model, which FR3 forbids, so verdicts ride on the text instead.
    """
    agent: Agent[EpisodeDeps, str] = Agent(
        model,
        deps_type=EpisodeDeps,
        output_type=str,
        instructions=instructions,
        tools=[Tool(get_series, takes_ctx=True), Tool(get_pool_meta_data, takes_ctx=True)],
        model_settings=model_settings,
        retries={"tools": tool_retries, "output": output_retries},
        name=name,
    )

    @agent.output_validator
    def _every_examined_series_has_a_verdict(ctx: RunContext[EpisodeDeps], output: str) -> str:
        deps = ctx.deps
        # The runner ingests each response as it arrives; this is the fallback for a plain
        # ``agent.run`` and a no-op otherwise (ingestion is idempotent per request).
        deps.collector.ingest(output, ctx.run_step, deps.pool)
        if not deps.enforce_verdicts:
            return output
        missing = deps.collector.missing_verdicts()
        if not missing:
            return output
        last = deps.collector.results.get(ctx.run_step)
        raise ModelRetry(retry_message(missing, last.rejected if last else []))

    return agent


def retry_message(missing: list[str], rejected: list[tuple[ParsedVerdict, str]], *, max_ids: int = 20) -> str:
    shown = ", ".join(missing[:max_ids])
    more = f" (and {len(missing) - max_ids} more)" if len(missing) > max_ids else ""
    text = (
        f"Your reply would end the episode, but {len(missing)} retrieved series still have no verdict: "
        f"{shown}{more}. Reply with one verdict line per series, exactly "
        '{"series_id": "<id>", "verdict": "normal"} or '
        '{"series_id": "<id>", "verdict": "anomalous", "explanation": "<evidence>"}, '
        "then the DONE line."
    )
    if rejected:
        notes = "; ".join(f"{cand.series_id} ({reason})" for cand, reason in rejected[:max_ids])
        text += f" Verdicts in your last message that were discarded: {notes}."
    return text


def registered_tool_names(agent: Agent[Any, Any]) -> list[str]:
    """Names of every function tool registered with ``agent`` (output tools excluded)."""
    names: list[str] = []
    for toolset in agent.toolsets:
        tools = getattr(toolset, "tools", None)
        if isinstance(tools, dict):
            names.extend(tools.keys())
    return names
