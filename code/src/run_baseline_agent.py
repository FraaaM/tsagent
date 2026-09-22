"""Run the raw-retrieval baseline agent over one pool (spec 02, section 6.2, step 4).

Configuration-driven, like the other scripts of this repository: every knob lives in the
CONFIGURATION block below and there are no command-line arguments. Edit the block, run
the file from anywhere:

    python code/src/run_baseline_agent.py

One run writes, under ``OUTPUT_DIR / RUN_NAME``:

    anomaly_verdicts.jsonl   one record per examined series (FR5)
    decision_trace.jsonl     tool calls, model requests, per-series roll-ups, totals (FR6)
    messages.json            the conversation as pydantic-ai kept it
    summary.json             the run_end record plus run identifiers
    series_aliases.json      alias -> pool id, only when ALIAS_SERIES_IDS is on

Both output files are validated after the run; a broken invariant fails the script. With
``REPLAYS > 1`` the same episode is run again into sibling directories and every replay is
compared with the first, field for field, which is the FR7 acceptance check.

``MODEL = "scripted"`` needs no API: it swaps the LLM for a deterministic stand-in that
walks the served order, retrieves each series once and judges it with a fixed rule, so the
whole pipeline can be exercised on a real pool at no cost. Its verdicts mean nothing for
the research questions.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_HERE.parents[1]))

from src.baseline_agent.models import model_slug  # noqa: E402
from src.baseline_agent.pool import Pool, read_split_ids  # noqa: E402
from src.baseline_agent.runner import TRACE_FILE, VERDICTS_FILE, EpisodeConfig, compare_runs, run_episode  # noqa: E402
from src.baseline_agent.schemas import validate_trace_file, validate_verdicts_file  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("run_baseline_agent")


# ======================================================================================
# CONFIGURATION - every knob for a run lives in this block
# ======================================================================================

# --- Pool ---
# The spec-01 pools live under data/<task 01>/; the DVC checkout names the directory
# "anomaly_detection", a local rebuild names it after the task. Both are tried in order.
_DATA_DIRS = [
    _REPO / "data" / "01-anomaly-detection-agent-spec",
    _REPO / "data" / "anomaly_detection",
]
POOL_FILE = "real/R1.parquet"     # relative to the first existing _DATA_DIRS entry, or absolute
SPLIT = None                      # "dev" / "test" to restrict to one split via the
                                  # sibling *_metadata.parquet, None for the whole pool
ORDER_SEED = 42                   # seeded permutation of the order the agent is shown
                                  # (spec-01 files are grouped by source, so file order
                                  # walks one recording at a time); None = file order
SERIES_LIMIT = None               # keep only the first N series of the served order; None = all
ALIAS_SERIES_IDS = False          # True: the agent sees opaque ids (s0001, ...) instead of
                                  # the provenance-carrying spec-01 ids; outputs keep real ids

# --- Model ---
# The constitution caps the LLM at ~20B parameters, so the intended targets are local or
# hosted open models: "openai-chat:gpt-oss-20b" with MODEL_BASE_URL pointing at vLLM /
# Ollama / LM Studio, "ollama:qwen3:14b", "openrouter:openai/gpt-oss-20b". Any other
# pydantic-ai model string works for harness checks. "scripted" = dry run without an LLM.
MODEL = "scripted"
MODEL_BASE_URL = None             # OpenAI-compatible endpoint, e.g. "http://localhost:8000/v1"
MODEL_API_KEY_ENV = "OPENAI_API_KEY"   # environment variable read for MODEL_BASE_URL endpoints
                                  # (provider strings use their own: OPENROUTER_API_KEY, ...)
SCRIPTED_OPTIONS = {"max_series": 25}  # stand-in knobs; ignored for real models

# --- Decoding (FR7: greedy, seeded, one tool call per turn) ---
TEMPERATURE = 0.0
SEED = 42                         # honoured by OpenAI-compatible servers; ignored elsewhere
MAX_TOKENS = None                 # per-response cap; None = provider default
PARALLEL_TOOL_CALLS = False       # one retrieval per turn (spec 02, idea B); None = provider default
EXTRA_MODEL_SETTINGS = {}         # provider-specific pydantic-ai settings, e.g.
                                  # {"openai_reasoning_effort": "low"}

# --- Episode caps ---
# pydantic-ai checks these after each response; the harness additionally projects the
# next request from the payloads the model just asked for and stops *before* it when the
# projection breaks TOTAL_TOKENS_LIMIT or CONTEXT_TOKENS_LIMIT (termination "budget").
REQUEST_LIMIT = None              # model requests per episode; None = 2 per series + 8
TOOL_CALLS_LIMIT = None           # executed tool calls per episode; None = 2 per series + 8
TOTAL_TOKENS_LIMIT = 1_000_000    # constitution budget: about one day of API access
CONTEXT_TOKENS_LIMIT = None       # the model's window, e.g. 128_000; None = provider decides
CHARS_PER_TOKEN_ESTIMATE = 1.5    # for the projection; numeric JSON measures 1.6-2.2
OUTPUT_ALLOWANCE_TOKENS = 1024    # reply size assumed by the projection when MAX_TOKENS is None
TOOL_RETRIES = 1                  # retries pydantic-ai grants on invalid tool arguments
OUTPUT_RETRIES = 2                # nudges when the agent stops with verdicts missing
ENFORCE_VERDICTS = True           # False: let the agent stop with examined-but-unjudged series
VALUE_PRECISION = None            # decimals to round values to in get_series; None = raw

# --- Output ---
OUTPUT_DIR = _REPO / "data" / "02-raw-retrieval-baseline-agent-infrastructure" / "runs"
RUN_NAME = None                   # None = "<pool>[_<split>][_first<N>]__<model slug>"
REPLAYS = 1                       # > 1 runs the episode again and checks FR7 determinism
OVERWRITE = False                 # True: reuse an existing run directory


# ======================================================================================


def resolve_pool_file() -> Path:
    path = Path(POOL_FILE)
    if path.is_absolute():
        return path
    for base in _DATA_DIRS:
        if base.is_dir():
            return base / path
    raise FileNotFoundError(f"no pool directory found; tried {[str(p) for p in _DATA_DIRS]} - run `dvc pull` first")


def load_pool() -> Pool:
    pool_file = resolve_pool_file()
    logger.info("loading pool %s", pool_file)
    pool = Pool.from_parquet(pool_file)
    if SPLIT:
        meta = pool_file.with_name(f"{pool_file.stem}_metadata.parquet")
        pool = pool.subset(read_split_ids(meta, SPLIT))
        logger.info("split %r: %d series", SPLIT, len(pool))
    if SERIES_LIMIT is not None:
        # The limit applies to the served order, so a seeded order gives a seeded sample.
        pool = pool.reordered(ORDER_SEED)
        pool = pool.subset(pool.series_ids[:SERIES_LIMIT])
        logger.info("first %d series of the served order kept", len(pool))
    return pool


def build_config() -> EpisodeConfig:
    return EpisodeConfig(
        model=MODEL,
        temperature=TEMPERATURE,
        seed=SEED,
        max_tokens=MAX_TOKENS,
        parallel_tool_calls=PARALLEL_TOOL_CALLS,
        extra_model_settings=dict(EXTRA_MODEL_SETTINGS),
        order_seed=ORDER_SEED,
        alias_series_ids=ALIAS_SERIES_IDS,
        request_limit=REQUEST_LIMIT,
        tool_calls_limit=TOOL_CALLS_LIMIT,
        total_tokens_limit=TOTAL_TOKENS_LIMIT,
        context_tokens_limit=CONTEXT_TOKENS_LIMIT,
        chars_per_token_estimate=CHARS_PER_TOKEN_ESTIMATE,
        output_allowance_tokens=OUTPUT_ALLOWANCE_TOKENS,
        tool_retries=TOOL_RETRIES,
        output_retries=OUTPUT_RETRIES,
        enforce_verdicts=ENFORCE_VERDICTS,
        value_precision=VALUE_PRECISION,
        provider_base_url=MODEL_BASE_URL,
        scripted_options=dict(SCRIPTED_OPTIONS),
    )


def run_name(pool: Pool) -> str:
    if RUN_NAME:
        return RUN_NAME
    parts = [pool.name]
    if SPLIT:
        parts.append(SPLIT)
    if SERIES_LIMIT is not None:
        parts.append(f"first{SERIES_LIMIT}")
    return "_".join(parts) + "__" + model_slug(MODEL)


def main() -> int:
    pool = load_pool()
    config = build_config()
    api_key = os.environ.get(MODEL_API_KEY_ENV) if MODEL_BASE_URL else None
    name = run_name(pool)
    results = []
    exit_code = 0
    for replay in range(1, REPLAYS + 1):
        out_dir = OUTPUT_DIR / (name if replay == 1 else f"{name}__replay{replay}")
        logger.info("episode %d/%d -> %s", replay, REPLAYS, out_dir)
        result = run_episode(pool, config, out_dir, api_key=api_key, overwrite=OVERWRITE)
        results.append(result)
        end = result.run_end
        logger.info(
            "termination=%s requests=%d get_series=%d examined=%d verdicted=%d anomalous=%d "
            "tokens=%d (in %d / out %d) wall=%.1fs",
            end.termination.reason,
            end.requests,
            end.tool_calls.get_series,
            end.series.examined,
            end.series.verdicted,
            end.series.anomalous,
            end.usage.total_tokens,
            end.usage.input_tokens,
            end.usage.output_tokens,
            end.duration_ms / 1000.0,
        )
        if end.termination.detail:
            logger.info("termination detail: %s", end.termination.detail)
        try:
            validate_verdicts_file(result.files[VERDICTS_FILE])
            validate_trace_file(result.files[TRACE_FILE])
        except ValueError as exc:
            logger.error("output validation failed: %s", exc)
            exit_code = 1
        if end.termination.reason == "interrupted":
            logger.warning("interrupted - not starting further replays")
            exit_code = 130
            break

    for replay, result in enumerate(results[1:], start=2):
        diff = compare_runs(results[0].output_dir, result.output_dir)
        if diff.identical:
            logger.info("replay %d: identical to the first run (timing, generated ids and cache accounting aside)", replay)
        else:
            exit_code = exit_code or 1
            logger.error("replay %d: differs from the first run:", replay)
            for line in diff.differences:
                logger.error("  %s", line)
    for result in results:
        logger.info("outputs: %s", result.output_dir)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
