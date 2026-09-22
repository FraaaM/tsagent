"""One episode, end to end: agent loop, verdict capture, trace, output files (spec 02, §5).

``run_episode`` drives the pydantic-ai graph step by step (``agent.iter``) so that every
model round trip can be timed and its usage recorded, and so that the verdict lines in each
response are captured the moment they arrive - not only at the end. Whatever ends the
episode (the agent stopping, a usage limit, a provider error, Ctrl-C), the outputs are
closed properly: ``anomaly_verdicts.jsonl`` holds the standing verdicts,
``decision_trace.jsonl`` ends with a ``run_end`` record naming the termination reason,
``messages.json`` holds the full conversation, ``summary.json`` the headline numbers.

``compare_runs`` is the FR7 check: two output directories are equal field-for-field once
wall-clock timing and provider-generated ids are set aside.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pydantic_ai
from pydantic_ai import Agent, UsageLimits
from pydantic_ai import messages as M
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from .agent import TOOL_NAMES, EpisodeDeps, build_agent, registered_tool_names
from .models import build_model
from .pool import Pool
from .prompt import SYSTEM_PROMPT, USER_PROMPT, prompt_fingerprint
from .schemas import (
    LimitsInfo,
    ModelInfo,
    PoolInfo,
    PromptInfo,
    RetriesInfo,
    RunEndRecord,
    RunStartRecord,
    Termination,
    UsageFields,
    read_jsonl,
    strip_timing,
)
from .trace import TraceRecorder, now_iso
from .verdicts import VerdictCollector
from .writers import JsonlWriter, VerdictWriter, write_json

logger = logging.getLogger(__name__)

VERDICTS_FILE = "anomaly_verdicts.jsonl"
TRACE_FILE = "decision_trace.jsonl"
MESSAGES_FILE = "messages.json"
SUMMARY_FILE = "summary.json"

# Fields that legitimately differ between two replays of the same configuration.
NONDETERMINISTIC_TRACE_FIELDS = frozenset({"tool_call_id"})
NONDETERMINISTIC_MESSAGE_FIELDS = frozenset(
    {"timestamp", "tool_call_id", "run_id", "conversation_id", "provider_response_id", "provider_details", "provider_name", "provider_url", "id"}
)


@dataclass(frozen=True)
class EpisodeConfig:
    """Everything that determines an episode's behaviour besides the pool itself."""

    model: str
    temperature: float | None = 0.0
    seed: int | None = 42
    max_tokens: int | None = None
    parallel_tool_calls: bool | None = None
    extra_model_settings: dict[str, Any] = field(default_factory=dict)
    request_limit: int | None = None  # None: 2 requests per series + 8
    tool_calls_limit: int | None = None
    total_tokens_limit: int | None = 1_000_000
    tool_retries: int = 1
    output_retries: int = 2
    enforce_verdicts: bool = True
    value_precision: int | None = None
    system_prompt: str = SYSTEM_PROMPT
    user_prompt: str = USER_PROMPT
    provider_base_url: str | None = None
    scripted_options: dict[str, Any] = field(default_factory=dict)

    def model_settings(self) -> ModelSettings | None:
        settings: dict[str, Any] = {}
        if self.temperature is not None:
            settings["temperature"] = self.temperature
        if self.seed is not None:
            settings["seed"] = self.seed
        if self.max_tokens is not None:
            settings["max_tokens"] = self.max_tokens
        if self.parallel_tool_calls is not None:
            settings["parallel_tool_calls"] = self.parallel_tool_calls
        settings.update(self.extra_model_settings)
        return ModelSettings(**settings) if settings else None  # type: ignore[typeddict-item]

    def effective_request_limit(self, n_series: int) -> int | None:
        return self.request_limit if self.request_limit is not None else 2 * n_series + 8

    def usage_limits(self, n_series: int) -> UsageLimits:
        return UsageLimits(
            request_limit=self.effective_request_limit(n_series),
            tool_calls_limit=self.tool_calls_limit,
            total_tokens_limit=self.total_tokens_limit,
        )

    def fingerprint(self) -> str:
        """Identity of the configuration - excludes the endpoint address, which is deployment, not behaviour."""
        payload = asdict(self)
        payload.pop("provider_base_url")
        payload["system_prompt"] = prompt_fingerprint(self.system_prompt)
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


@dataclass
class EpisodeResult:
    output_dir: Path
    run_id: str
    termination: Termination
    run_end: RunEndRecord
    files: dict[str, Path]

    @property
    def usage(self) -> UsageFields:
        return self.run_end.usage


# --- the episode ------------------------------------------------------------------------


def run_episode(
    pool: Pool,
    config: EpisodeConfig,
    output_dir: str | Path,
    *,
    model: Model | None = None,
    api_key: str | None = None,
    overwrite: bool = False,
) -> EpisodeResult:
    """Run one episode over ``pool`` and write its outputs into ``output_dir``."""
    return asyncio.run(run_episode_async(pool, config, output_dir, model=model, api_key=api_key, overwrite=overwrite))


async def run_episode_async(
    pool: Pool,
    config: EpisodeConfig,
    output_dir: str | Path,
    *,
    model: Model | None = None,
    api_key: str | None = None,
    overwrite: bool = False,
) -> EpisodeResult:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir} (pass overwrite=True to reuse it)")
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {name: output_dir / name for name in (VERDICTS_FILE, TRACE_FILE, MESSAGES_FILE, SUMMARY_FILE)}

    if model is None:
        model = build_model(config.model, base_url=config.provider_base_url, api_key=api_key, scripted_options=config.scripted_options)
    agent = build_agent(
        model,
        model_settings=config.model_settings(),
        instructions=config.system_prompt,
        tool_retries=config.tool_retries,
        output_retries=config.output_retries,
    )
    tools = registered_tool_names(agent)
    if sorted(tools) != sorted(TOOL_NAMES):
        raise RuntimeError(f"agent exposes tools {tools}, expected exactly {list(TOOL_NAMES)}")

    run_id = hashlib.sha256((pool.fingerprint + config.fingerprint()).encode()).hexdigest()[:12]
    collector = VerdictCollector()
    verdict_writer = VerdictWriter(files[VERDICTS_FILE])
    trace_writer = JsonlWriter(files[TRACE_FILE])
    recorder = TraceRecorder(trace_writer, pool)
    deps = EpisodeDeps(pool=pool, recorder=recorder, collector=collector, value_precision=config.value_precision, enforce_verdicts=config.enforce_verdicts)
    limits = config.usage_limits(len(pool))

    recorder.start(
        RunStartRecord(
            run_id=run_id,
            config_fingerprint=config.fingerprint(),
            pool=PoolInfo(name=pool.name, path=pool.source, n_series=len(pool), fingerprint=pool.fingerprint),
            model=ModelInfo(name=config.model, settings=dict(config.model_settings() or {}), provider_base_url=config.provider_base_url),
            usage_limits=LimitsInfo(request_limit=limits.request_limit, tool_calls_limit=limits.tool_calls_limit, total_tokens_limit=limits.total_tokens_limit),
            retries=RetriesInfo(tools=config.tool_retries, output=config.output_retries),
            prompt=PromptInfo(system_prompt_sha256=prompt_fingerprint(config.system_prompt), user_prompt=config.user_prompt),
            tools=tools,
            value_precision=config.value_precision,
            harness={"pydantic_ai": pydantic_ai.__version__, "python": platform.python_version(), "platform": platform.platform()},
            started_at=now_iso(),
        )
    )
    logger.info("run %s: pool %s (%d series), model %s, request limit %s", run_id, pool.name, len(pool), config.model, limits.request_limit)

    try:
        termination, messages = await _drive(agent, deps, config, limits, verdict_writer)
    except BaseException:
        # Anything _drive did not translate into a termination (SystemExit, task cancellation)
        # still leaves a closed trace behind.
        recorder.finish(Termination(reason="interrupted", detail="aborted before completion"), collector)
        trace_writer.close()
        raise
    run_end = recorder.finish(termination, collector)
    trace_writer.close()
    write_json(files[MESSAGES_FILE], ModelMessagesTypeAdapter.dump_python(messages, mode="json"))
    summary = {
        "run_id": run_id,
        "pool": pool.name,
        "model": config.model,
        "output_dir": str(output_dir),
        **run_end.model_dump(mode="json"),
    }
    write_json(files[SUMMARY_FILE], summary)
    logger.info(
        "run %s finished: %s; %d requests, %d get_series calls, %d/%d series verdicted (%d anomalous), %d tokens",
        run_id,
        termination.reason,
        run_end.requests,
        run_end.tool_calls.get_series,
        run_end.series.verdicted,
        len(pool),
        run_end.series.anomalous,
        run_end.usage.total_tokens,
    )
    return EpisodeResult(output_dir=output_dir, run_id=run_id, termination=termination, run_end=run_end, files=files)


async def _drive(
    agent: Agent[EpisodeDeps, str],
    deps: EpisodeDeps,
    config: EpisodeConfig,
    limits: UsageLimits,
    verdict_writer: VerdictWriter,
) -> tuple[Termination, list[M.ModelMessage]]:
    """Step through the agent graph, recording each model request and ingesting its text."""
    recorder, collector, pool = deps.recorder, deps.collector, deps.pool
    request_index = 0
    termination = Termination(reason="completed")
    run: Any = None
    pending: dict[str, Any] | None = None  # timing of a request in flight

    def _record_response(response: M.ModelResponse, *, started_at: str, duration_ms: float, is_retry: bool, history_chars: int) -> None:
        text = "\n".join(p.content for p in response.parts if isinstance(p, M.TextPart))
        tool_parts = [p for p in response.parts if isinstance(p, M.ToolCallPart)]
        ingest = collector.ingest(text, request_index, pool)
        for rec in ingest.accepted:
            verdict_writer.put(rec)
        requested = [str(_args_as_dict(p).get("series_id")) for p in tool_parts if p.tool_name == "get_series"]
        recorder.model_request(
            request_index=request_index,
            is_retry=is_retry,
            started_at=started_at,
            duration_ms=duration_ms,
            usage=response.usage,
            model_name=response.model_name,
            finish_reason=response.finish_reason,
            history_chars=history_chars,
            response_text_chars=len(text),
            n_tool_calls=len(tool_parts),
            requested_series=requested,
            verdicts_accepted=len(ingest.accepted),
            verdicts_rejected=len(ingest.rejected),
        )
        recorder.verdicts(ingest.events)
        logger.info(
            "request %d: in=%s out=%s tokens, context=%d chars, %d tool call(s) %s, verdicts +%d/-%d (%d standing)",
            request_index,
            response.usage.input_tokens,
            response.usage.output_tokens,
            history_chars,
            len(tool_parts),
            requested[:3],
            len(ingest.accepted),
            len(ingest.rejected),
            len(collector.final),
        )

    try:
        async with agent.iter(config.user_prompt, deps=deps, usage_limits=limits) as run:
            node = run.next_node
            while not Agent.is_end_node(node):
                if Agent.is_model_request_node(node):
                    request_index += 1
                    history = list(run.all_messages())
                    if not history or history[-1] is not node.request:
                        history.append(node.request)
                    pending = {
                        "is_retry": any(isinstance(p, M.RetryPromptPart) for p in node.request.parts),
                        "history_chars": history_chars(history, instructions=config.system_prompt),
                        "started_at": now_iso(),
                        "t0": time.perf_counter(),
                    }
                    next_node = await run.next(node)
                    duration_ms = (time.perf_counter() - pending["t0"]) * 1000.0
                    if Agent.is_call_tools_node(next_node):
                        _record_response(
                            next_node.model_response,
                            started_at=pending["started_at"],
                            duration_ms=duration_ms,
                            is_retry=pending["is_retry"],
                            history_chars=pending["history_chars"],
                        )
                    pending = None
                    node = next_node
                elif Agent.is_call_tools_node(node):
                    async with node.stream(run.ctx) as stream:
                        async for event in stream:
                            if isinstance(event, M.FunctionToolResultEvent) and isinstance(event.part, M.RetryPromptPart):
                                recorder.invalid_tool_call(
                                    event.part.tool_name or "?",
                                    event.part.tool_call_id,
                                    _args_of(node.model_response, event.part.tool_call_id),
                                    event.part.model_response(),
                                    request_index=request_index,
                                )
                    node = await run.next(node)
                else:
                    node = await run.next(node)
    except UsageLimitExceeded as exc:
        termination = Termination(reason="usage_limit", detail=_short(exc))
    except UnexpectedModelBehavior as exc:
        termination = Termination(reason="protocol_error", detail=_short(exc))
    except KeyboardInterrupt:
        termination = Termination(reason="interrupted", detail="KeyboardInterrupt")
    except Exception as exc:  # provider/API failures: recorded, not swallowed silently
        logger.exception("episode aborted during request %d", request_index)
        termination = Termination(reason="model_error", detail=_short(exc))

    messages = list(run.all_messages()) if run is not None else []
    if run is not None and pending is not None:
        # A limit that trips *after* a response was received (total_tokens_limit) leaves
        # the run's usage one request ahead of the trace. Book that response - from the
        # history when pydantic-ai kept it, from the usage totals when it did not - so the
        # trace never under-reports what the provider charged.
        responses = [m for m in messages if isinstance(m, M.ModelResponse)]
        duration_ms = (time.perf_counter() - pending["t0"]) * 1000.0
        if len(responses) > len(recorder.requests):
            _record_response(responses[-1], started_at=pending["started_at"], duration_ms=duration_ms, is_retry=pending["is_retry"], history_chars=pending["history_chars"])
        elif run.usage.requests > len(recorder.requests):
            recorder.model_request(
                request_index=request_index,
                is_retry=pending["is_retry"],
                started_at=pending["started_at"],
                duration_ms=duration_ms,
                usage=_usage_delta(run.usage, recorder.requests),
                model_name=None,
                finish_reason=None,
                history_chars=pending["history_chars"],
                response_text_chars=0,
                n_tool_calls=0,
                requested_series=[],
                verdicts_accepted=0,
                verdicts_rejected=0,
            )
            termination = Termination(
                reason=termination.reason,
                detail=f"{termination.detail} [the last response was charged but not retained by the harness; its usage is booked from the run totals]",
            )
    return termination, messages


# --- helpers ----------------------------------------------------------------------------


def history_chars(messages: Sequence[M.ModelMessage], *, instructions: str | None = None) -> int:
    """Size in characters of everything the model is sent - a provider-independent context measure.

    ``instructions`` is counted once when no request in ``messages`` carries them yet
    (pydantic-ai attaches the system prompt while preparing the first request).
    """
    total = 0
    carried = False
    for msg in messages:
        if isinstance(msg, M.ModelRequest) and msg.instructions:
            total += len(msg.instructions)
            carried = True
        for part in msg.parts:
            if isinstance(part, M.ToolReturnPart):
                total += len(part.model_response_str())
            elif isinstance(part, M.ToolCallPart):
                total += len(part.tool_name) + len(part.args_as_json_str())
            elif isinstance(part, M.RetryPromptPart):
                total += len(part.model_response())
            else:
                content = getattr(part, "content", "")
                total += len(content) if isinstance(content, str) else len(json.dumps(content, default=str))
    if instructions and not carried:
        total += len(instructions)
    return total


def _usage_delta(total: Any, recorded: Sequence[Any]) -> UsageFields:
    """Run usage minus what the trace has booked so far - the share of the unrecorded response."""
    return UsageFields(
        input_tokens=int(total.input_tokens) - sum(r.usage.input_tokens for r in recorded),
        output_tokens=int(total.output_tokens) - sum(r.usage.output_tokens for r in recorded),
        cache_read_tokens=int(total.cache_read_tokens) - sum(r.usage.cache_read_tokens for r in recorded),
        cache_write_tokens=int(total.cache_write_tokens) - sum(r.usage.cache_write_tokens for r in recorded),
    )


def _args_as_dict(part: M.ToolCallPart) -> dict[str, Any]:
    try:
        args = part.args_as_dict()
    except Exception:
        return {"raw": part.args}
    return args if isinstance(args, dict) else {"raw": args}


def _args_of(response: M.ModelResponse, tool_call_id: str | None) -> Any:
    for part in response.parts:
        if isinstance(part, M.ToolCallPart) and part.tool_call_id == tool_call_id:
            return _args_as_dict(part)
    return {}


def _short(exc: BaseException, limit: int = 500) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= limit else text[: limit - 3] + "..."


# --- FR7: replay comparison -------------------------------------------------------------


@dataclass
class RunDiff:
    identical: bool
    differences: list[str]


def _strip(obj: Any, fields: frozenset[str]) -> Any:
    if isinstance(obj, dict):
        return {k: _strip(v, fields) for k, v in obj.items() if k not in fields}
    if isinstance(obj, list):
        return [_strip(v, fields) for v in obj]
    return obj


def _first_difference(a: list[Any], b: list[Any], label: str) -> str | None:
    if len(a) != len(b):
        return f"{label}: {len(a)} vs {len(b)} records"
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            kind = x.get("kind", "?") if isinstance(x, dict) else "?"
            return f"{label}: record {i + 1} ({kind}) differs: {json.dumps(x, ensure_ascii=False)[:200]} != {json.dumps(y, ensure_ascii=False)[:200]}"
    return None


def compare_runs(dir_a: str | Path, dir_b: str | Path, *, compare_messages: bool = True) -> RunDiff:
    """Field-for-field equality of two episode outputs, ignoring timing and generated ids."""
    dir_a, dir_b = Path(dir_a), Path(dir_b)
    diffs: list[str] = []
    d = _first_difference(read_jsonl(dir_a / VERDICTS_FILE), read_jsonl(dir_b / VERDICTS_FILE), VERDICTS_FILE)
    if d:
        diffs.append(d)
    trace_a = _strip(strip_timing(read_jsonl(dir_a / TRACE_FILE)), NONDETERMINISTIC_TRACE_FIELDS)
    trace_b = _strip(strip_timing(read_jsonl(dir_b / TRACE_FILE)), NONDETERMINISTIC_TRACE_FIELDS)
    d = _first_difference(trace_a, trace_b, TRACE_FILE)
    if d:
        diffs.append(d)
    if compare_messages:
        msgs_a = _strip(json.loads((dir_a / MESSAGES_FILE).read_text(encoding="utf-8")), NONDETERMINISTIC_MESSAGE_FIELDS)
        msgs_b = _strip(json.loads((dir_b / MESSAGES_FILE).read_text(encoding="utf-8")), NONDETERMINISTIC_MESSAGE_FIELDS)
        d = _first_difference(msgs_a, msgs_b, MESSAGES_FILE)
        if d:
            diffs.append(d)
    return RunDiff(identical=not diffs, differences=diffs)
