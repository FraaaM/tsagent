# 02 - Raw Retrieval Baseline Agent Infrastructure

The raw-retrieval-only baseline agent of RQ1: one pydantic-ai agent with exactly two tools,
`get_series` and `get_pool_meta_data`, that works through a pool in a single episode and
emits a verdict for every series it examines. It is the reference point the tool-augmented
agent of M2/M3 is compared against, and the reference point of RQ3's context-efficiency
measurements, so its token consumption is logged down to the request.

The contract is
[`.internal/specs/02-raw-retrieval-baseline-agent-infrastructure-spec.md`](../../.internal/specs/02-raw-retrieval-baseline-agent-infrastructure-spec.md)
(issue [dynolab/tsagent#2](https://github.com/dynolab/tsagent/issues/2)). As the spec
asks, the code lives in [`code/`](../../code/), not here; this file covers what is done,
how to run it and what to expect from it.

## Status

- [x] Pool storage and the two tools (FR1, FR2) -
      [`pool.py`](../../code/src/baseline_agent/pool.py),
      [`tools.py`](../../code/src/baseline_agent/tools.py). The `label` column is never
      read; unknown or malformed ids come back as an error sentence, not an exception.
- [x] The agent (FR3, spec §4.2): two function tools and nothing else, plain-text output,
      greedy decoding, an efficiency-focused system prompt with the stop-when-confident
      rule - [`agent.py`](../../code/src/baseline_agent/agent.py),
      [`prompt.py`](../../code/src/baseline_agent/prompt.py).
- [x] One episode per pool with the agent choosing what to examine (FR4), the verdict
      protocol and writer (FR5), the decision trace with per-call timing and per-request
      token usage (FR6, NFR) -
      [`runner.py`](../../code/src/baseline_agent/runner.py),
      [`verdicts.py`](../../code/src/baseline_agent/verdicts.py),
      [`trace.py`](../../code/src/baseline_agent/trace.py),
      [`writers.py`](../../code/src/baseline_agent/writers.py).
- [x] Entry point [`run_baseline_agent.py`](../../code/src/run_baseline_agent.py) and the
      replay comparison behind FR7 (`REPLAYS > 1`).
- [x] Unit tests, no LLM and no GPU (spec §3, unit level): tool registration, tool
      behaviour, output schemas, and whole episodes driven by a scripted stand-in model,
      including every termination path - [`code/tests/`](../../code/tests/), 59 tests.
- [x] Dry runs on the real R1 pool with the stand-in: pipeline, trace consistency and
      replay identity hold on real data.
- [ ] End-to-end runs with an LLM (spec §3, E2E level): a replay pair for FR7 and a full
      run over R1 or R2 within the token budget. Blocked on an API endpoint or a local
      model server; nothing in the harness needs to change to do them, see
      [Running with a real model](#running-with-a-real-model) and
      [What to expect from a full run](#what-to-expect-from-a-full-run).

## Layout

```
code/src/baseline_agent/
  pool.py        Pool: label-free view of a spec-01 parquet file, series in file order
  tools.py       get_series / get_pool_meta_data as pure functions returning text
  prompt.py      system prompt (budget instruction, verdict format, stop rule), user prompt
  agent.py       pydantic-ai Agent: the two tools with timing/trace wrappers, output check
  verdicts.py    verdict-line parser and the episode rules (examined-only, last wins)
  trace.py       decision-trace recorder, per-series roll-ups, token attribution
  writers.py     JSONL writers (verdicts file keeps one record per series)
  runner.py      run_episode: agent.iter() loop, termination handling, compare_runs (FR7)
  models.py      model string -> pydantic-ai Model (providers, OpenAI-compatible endpoints)
  scripted.py    deterministic stand-in for the LLM - dry runs and tests only
code/src/run_baseline_agent.py   configuration-driven entry point
code/tests/                      test_tools.py, test_output_schemas.py, test_runner.py
(runs land in ../../data/02-raw-retrieval-baseline-agent-infrastructure/runs/, git-ignored)
```

## Running

Install once and make sure the pools are in place (`dvc pull`, or rebuild them with the
task-01 scripts):

```bash
pip install -r code/requirements.txt
```

The entry point follows the repository convention: every knob sits in the `CONFIGURATION`
block at the top of the file, there are no command-line arguments. Edit the block, run the
file:

```bash
python code/src/run_baseline_agent.py
```

The defaults make a **dry run**: `MODEL = "scripted"` replaces the LLM with a
deterministic stand-in that walks the first 25 series of R1 in pool order, retrieves each
once, judges it with a fixed rule and stops. It exercises everything except the model and
needs no API. Its verdicts say nothing about anomaly detection.

Useful knobs:

| Knob | Meaning |
|------|---------|
| `POOL_FILE`, `SPLIT`, `SERIES_LIMIT` | which pool, optionally one split (`dev`/`test`) or the first N series |
| `MODEL`, `MODEL_BASE_URL`, `MODEL_API_KEY_ENV` | which LLM, and where (see below) |
| `TEMPERATURE = 0.0`, `SEED = 42` | greedy, seeded decoding (FR7) |
| `REQUEST_LIMIT`, `TOOL_CALLS_LIMIT`, `TOTAL_TOKENS_LIMIT` | the episode caps; the token cap defaults to the constitution's 1M |
| `OUTPUT_RETRIES`, `ENFORCE_VERDICTS` | the missing-verdict nudge (see design decisions) |
| `REPLAYS` | `2` runs the episode twice and reports whether the outputs are identical |

Tests:

```bash
cd code
python -m pytest
```

### Running with a real model

`MODEL` takes any pydantic-ai model string; credentials come from the provider's usual
environment variable:

```python
MODEL = "anthropic:claude-sonnet-4-5"      # ANTHROPIC_API_KEY
MODEL = "openai:gpt-4.1-mini"              # OPENAI_API_KEY
MODEL = "openrouter:openai/gpt-oss-20b"    # OPENROUTER_API_KEY
```

A local model on the project GPU is served through an OpenAI-compatible endpoint
(vLLM, Ollama, LM Studio, llama.cpp all speak it):

```python
MODEL = "openai-chat:gpt-oss-20b"
MODEL_BASE_URL = "http://localhost:8000/v1"
MODEL_API_KEY_ENV = "OPENAI_API_KEY"       # whatever the server expects, or unset
```

Provider-specific settings go into `EXTRA_MODEL_SETTINGS` (for example
`{"openai_reasoning_effort": "low"}`). `SEED` is honoured by OpenAI-style chat endpoints
and vLLM and ignored by providers that do not expose one; pydantic-ai drops `temperature`
for the newest Claude models, which do not accept it. With `REPLAYS = 2` the script says
whether the two episodes agree, so the determinism a given endpoint actually delivers is
measured rather than assumed - hosted APIs do not promise bit-exact greedy decoding.

## Outputs

One directory per run, `OUTPUT_DIR / RUN_NAME` (default
`<pool>[_<split>][_first<N>]__<model slug>`, replays get a `__replay<k>` suffix):

**`anomaly_verdicts.jsonl`** (FR5) - one record per examined series, its standing verdict:

```json
{"series_id": "R1__YAHOO__real_42_full", "verdict": "normal"}
{"series_id": "R1__SMD__machine-1-1_col_3_chunk2", "verdict": "anomalous", "explanation": "values jump from about 0.02 to 0.9 between time_index 1180 and 1215"}
```

`explanation` is present exactly when the verdict is `anomalous`. No score, no rank.

**`decision_trace.jsonl`** (FR6) - chronological, one typed record per line, `kind` says which:

| `kind` | One per | Carries |
|--------|---------|---------|
| `run_start` | run | run id, pool (name, path, size, fingerprint), model and settings, limits, prompt hash, registered tools, harness versions |
| `model_request` | LLM round trip | index, retry flag, wall time, provider usage (input / output / cache tokens), context size in characters, tool calls requested, verdicts accepted / rejected |
| `tool_call` | tool invocation | tool, arguments, series id, ok or error text, points returned, payload size, wall time |
| `verdict` | parsed verdict | series id, verdict, whether the series was examined, accepted or the rejection reason, whether it superseded an earlier one |
| `series` | series touched | `get_series` call count (and failures), tool time, first / last request, standing verdict, revisions, attributed token usage |
| `run_end` | run | termination reason, totals (requests, retries, tool calls, tokens), overhead usage, coverage counts, verdict counts |

`termination.reason` is one of `completed` (the agent stopped by itself), `usage_limit`
(a cap tripped), `model_error` (the provider failed, e.g. context overflow),
`protocol_error` (the agent kept ending without verdicts for examined series) or
`interrupted`. Whatever the reason, the trace is closed by `run_end` and the verdicts
file holds every verdict accepted up to that point.

**`messages.json`** - the whole conversation as pydantic-ai recorded it, tool payloads
included: the ground truth for anything about context. **`summary.json`** - the
`run_end` record plus identifiers, for quick reading.

`baseline_agent.schemas.validate_verdicts_file` / `validate_trace_file` validate the two
files; `baseline_agent.runner.compare_runs` compares two run directories field for field,
ignoring wall-clock fields (`started_at`, `finished_at`, `duration_ms`, `tool_time_ms`)
and provider-generated ids.

## Design decisions worth knowing

**Verdicts ride on the agent's text.** FR3 allows exactly two tools and FR5 wants a
verdict per examined series, produced as the episode goes. An output tool would be a third
tool exposed to the model and would end the run on first use, so the agent writes each
verdict as a one-line JSON object in its reply and the harness parses every response the
moment it arrives. The parser is lenient about placement (bare lines, code fences,
pretty-printed objects, arrays, prose around them) and strict about content, which is the
`VerdictRecord` schema. Episode rules on top: a verdict counts only for a series whose
values the agent retrieved in this episode, unknown ids are discarded, the last verdict for
a series wins. Every accepted, revised or discarded verdict is a `verdict` record in the
trace, so hallucinated ids and blind verdicts are measured instead of silently dropped.

**One protocol check, no coverage push.** If the agent ends the episode while a series it
retrieved has no verdict, the output validator answers with a `ModelRetry` naming the
series (bounded by `OUTPUT_RETRIES`, counted in `run_end.retries`). That is the FR4
protocol - a verdict for each examined series - and nothing more: the harness never asks
the agent to examine another series, never re-fetches, never compacts the history.
Stopping early is the baseline's own property and stays visible as
`series.examined` / `series.in_pool`. `ENFORCE_VERDICTS = False` switches even that check
off.

**Label-free by construction.** `Pool.from_parquet` reads `series_id`, `time_index`,
`value` and nothing else; the metadata file is opened only to select a split, and only its
`series_id` / `split` columns. The prompt names no base rate. A test checks that neither
tool payload can carry a label.

**Tool results are text with a pinned wire format.** Both tools return the JSON string
the model will see (compact, no whitespace, insertion-ordered keys) rather than a dict for
pydantic-ai to serialise. That makes the payload size a fact (`payload_chars` in the
trace), keeps replays byte-identical across providers, and is where FR1's "textual error
message" lives: an unknown id or a broken series (NaN, non-contiguous index) yields
`Error: ...` for the model to read, never an exception.

**Every ending is a recorded ending.** The runner steps through pydantic-ai's graph
(`agent.iter`) so that each model request is timed and its usage booked as it happens,
and verdicts and trace lines are flushed to disk immediately. A usage cap, a provider
error, exhausted retries or Ctrl-C all end in a `run_end` record with the reason, and the
files written so far are complete as far as they go. A cap that trips after a response was
received is booked from the run's usage totals when pydantic-ai discards that response, so
the trace never under-reports what the provider charged.

**Token attribution.** Providers report usage per request, and one request can consume
the payloads of several series. Each request is charged to the series whose `get_series`
payloads it consumed (the successful calls executed right before it), split equally; the
prompt-only first turn, the metadata turn, the final message and retries are booked as
`overhead_usage`. Shares are fractional so that `series.attributed_usage` plus
`run_end.overhead_usage` equal `run_end.usage` exactly - a test checks the sum.

**Determinism (FR7).** Fixed prompt, pool order as in the file, greedy decoding with a
seed, no timestamps in anything the model sees, deterministic tool payloads. The run id is
a hash of the pool contents and the configuration, so two replays of the same setup share
it. What legitimately differs is wall-clock timing and provider-generated ids, and
`compare_runs` sets exactly those aside. `messages.json` is compared too, so the check
covers the whole conversation, not only the summaries.

**No history compaction, no batching, no rounding.** The natural fix for the context
arithmetic below - dropping old payloads, fetching several series per call, sending
fewer digits - would each change what the baseline is; spec 02 rejects batching (idea C)
and RQ3 is about tool outputs that avoid this cost in the first place. The harness keeps
every payload in context verbatim. `VALUE_PRECISION` exists for experiments on the last
point and is off by default.

**The scripted stand-in is a model, not a method.** `MODEL = "scripted"` plugs a
deterministic policy into pydantic-ai's `FunctionModel`: discover the pool, retrieve
series in order, judge each with a robust-z / frozen-run rule, stop. It also has switches
to misbehave on purpose (blind verdicts, unknown ids, malformed lines, never stopping,
skipping a verdict) so the tests can cover the harness's answers. Its token counts are
estimates (4 characters per token, optimistic for numeric JSON) and it is labelled
`scripted` everywhere.

## What to expect from a full run

`get_series` returns the whole series, and in a single episode every request re-sends the
whole history. Measured on the four pools (o200k tokenizer, the GPT-4o / gpt-oss family):

| Pool | Series | Points per series (median / max) | Payload per series, chars (median / max) | Chars per token | Tokens per series (median) | Whole pool retrieved once |
|------|--------|----------------------------------|------------------------------------------|-----------------|----------------------------|---------------------------|
| R1 | 2,900 | 1,500 / 12,960 | 24.5k / 231.6k | 1.73 | ~14.2k | ~55M tokens |
| R2 | 2,900 | 1,536 / 1,696 | 15.7k / 18.4k | 1.57 | ~10.0k | ~29M tokens |
| S1 | 2,000 | 2,519 / 4,000 | 59.1k / 97.4k | 2.23 | ~26.5k | ~53M tokens |
| S2 | 2,000 | 2,526 / 3,999 | 59.3k / 97.6k | 2.22 | ~26.7k | ~53M tokens |

Numeric JSON tokenizes at under two characters per token, and the synthetic pools carry
full-precision `float64` values (about 23 characters per point, twice the real pools).
After k series of R1 a request carries roughly 14k × k tokens and the episode has consumed
roughly 14k × k(k+1)/2:

- the constitution's budget of about 1M tokens is spent after ~11 R1 series (~14 on R2,
  ~8 on S1/S2);
- a 128k-token context window overflows after ~8 R1 series, and the largest R1 series
  alone (12,960 points, ~134k tokens) does not fit in it at all;
- the dry run with the stand-in (4 characters per token, so optimistic) reached the 1M cap
  after 20 R1 series and 21 requests, with the 20th series retrieved but not yet judged -
  reported as `examined_without_verdict = 1`.

So on this baseline "a full run over one real dataset" is a run that terminates cleanly
with every examined series judged and the caps respected, not one that reaches every
series: coverage of a 2,900-series pool is two orders of magnitude out of reach in a single
episode, which is precisely the property RQ1 and RQ3 are set up to quantify. The trace
records the coverage actually achieved and what it cost. For cheaper experiments,
`SPLIT = "dev"` and `SERIES_LIMIT` restrict the pool while keeping the episode's
single-invocation shape.
