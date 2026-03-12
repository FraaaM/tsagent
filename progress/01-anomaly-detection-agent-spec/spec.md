# Anomaly Detection Agent Spec

## One-Paragraph Spec
[TODO: Describe in 5–8 sentences what the MVP does end-to-end]

- **Input**: [TODO: dataset(s), format, pool size assumptions]
- **Output**: ranked list of anomalous series + human-readable explanation why each time series is deemed anomalous
- **Core idea**: LLM agent plans budgeted deep-dives using deterministic tools; narrative must be evidence-linked
- **Hard constraint**: agent must not make claims not supported by tool outputs

## Scope

### In scope (must-haves)
- Anomaly detection only
- Retrieve abnormal series from a pool of ~1,000 univariate series
- Full-series analysis (not streaming)
- Two-stage process:
  1) cheap scan across all series
  2) deep detect for a limited subset (budgeted)
- LLM tool-calling agent required: selects which series to deep-dive and which preset to use and then provides explanation for each anomalous time series
- Evaluation across 2 real industrial datasets + 2 synthetic datasets

### Out of scope (explicitly excluded)
- Streaming/online interface
- Computational issues
- Multivariate time series

## Datasets Commitment

### Real datasets (2)
| ID | Dataset name | Source / link | Unit of analysis | Sampling unit | Length (in sampling units) | Expected #series available | Notes / constraints |
|----|--------------|---------------|------------------|---------------|----------------------------|----------------------------|---------------------|
| R1 | TODO | TODO | univariate series | TODO | TODO | TODO | TODO |
| R2 | TODO | TODO | univariate series | TODO | TODO | TODO | TODO |

### Synthetic generators (2)
| ID | Generator name | Base process | Anomaly families injected | Controls (rate/severity/duration) |
|----|----------------|-------------|----------------------------|----------------------------------|
| S1 | TODO | TODO | TODO | TODO |
| S2 | TODO | TODO | TODO | TODO |

## Unified Data Format
Describe the on-disk format and how a deterministic ~1,000-series pool is created.

- **Canonical storage format**: [TODO: parquet/jsonl/etc.]
- **Series ID**: [TODO: stable ID rule]
- **Timestamps**: [TODO: timezone, monotonicity, missing handling]
- **Values**: [TODO: type, scaling, units]
- **Metadata**: [TODO: per-series fields]

## Label Definition (Series-Level)
This section must be precise and testable.

### What is the prediction target?
Define the label `y_i` for series `i`.

- **Label type**: binary series-level label `y_i ∈ {0,1}`
- **Positive class definition**: anomalous time series (what we call an anomaly depends on a particular problem and should be specified separately for each dataset)
- **Negative class definition**: non-anomalous time series
- **Ambiguous / unlabeled cases**: [TODO: drop? treat as negative? separate split?]

### How labels are derived (per dataset)
For each dataset, state the rule that transforms raw annotations/events into `y_i`.

- **R1 derivation**: [TODO: exact steps]
- **R2 derivation**: [TODO: exact steps]
- **S1 derivation**: injected anomaly => `y_i=1`
- **S2 derivation**: injected anomaly => `y_i=1`

### Known label noise / caveats
- [TODO]

## Tool Suite
Enumerate tools and their deterministic I/O at the level needed for evaluation.

### Cheap scan (global)

- [TODO: enumerate all the cheap tools following the format below]
- **Function**: `cheap_scan_all(pool) -> cheap_scan.json`
- **Per-series outputs** (minimum):
  - `series_id`
  - `cheap_score` (higher = more anomalous)
- **Determinism rule**: same pool + config => identical output

### Deep detect (per series)

- [TODO: enumerate all the deep detect tools following the format below]
- **Function**: `deep_detect(series, preset) -> deep_detect.json`
- **Per-series outputs** (minimum):
  - `series_id`
  - `deep_score`
  - `diagnostics` (structured fields only; no freeform)
  - `evidence`: pointers to time indices / segments / summary stats used to justify the score
- **Determinism rule**: same series + preset + config => identical output

## Agent Contract (Planning + Guardrails)
Define exactly what the agent is allowed to do and what it must output.

### Inputs available to the agent
- Cheap scan results (`cheap_scan.json`)
- Optional per-series metadata (define fields): [TODO]

### Budget model
Define a budget `B` deep-dives per 1,000 series.

- **B definition**: number of series allowed to call `deep_detect` on
- **Budgets to evaluate**: [TODO: e.g., B ∈ {0, 10, 25, 50, 100}]
- **Hard enforcement**: agent run must fail (or truncate) if budget exceeded

### Agent outputs (must be machine-checkable)
- `ranked_list.json` with (minimum):
  - ordered list of `series_id`
  - final anomaly score used for ranking
  - recommended threshold policy (dynamic K)
- `decision_trace.json` with (minimum):
  - which series were deep-dived
  - justification referencing only tool outputs (IDs/fields)

### Faithfulness / evidence requirements
- All natural-language claims must be traceable to structured tool outputs
- No claims about raw series unless supported by evidence fields returned by tools
- If evidence is insufficient, agent must say so and defer

## Baselines (Definitions + Expected Outputs)
Baselines must produce the same `ranked_list.json` schema.

- **Baseline A: cheap-only**
  - Ranking uses only `cheap_score` (plus any allowed deterministic post-processing)
- **Baseline B: random deep-dive**
  - Randomly select B series for deep detect, then rank using a fixed rule

For each baseline:
- **Tie-breaking rule**: [TODO]
- **Randomness control**: [TODO: seed policy]

## Metrics and Benchmark Reporting
Define metrics, how they’re computed, and what plots/tables are required.

### Ranking metrics
- AUROC for the varying threshold
- Adjusted Rand Index for the recommended threshold

### Budget / cost accounting
- Tool-call counts (cheap + deep)
- Runtime per tool and total runtime
- Quality-vs-budget curves: metric vs B

### Required benchmark report artifacts per run
- `metrics.json`
- single aggregated run summary (human-readable markdown): [TODO: filename]
