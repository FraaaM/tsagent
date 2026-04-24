# Anomaly Detection Agent Spec

## One-Paragraph Spec
The MVP functions as an offline agent that orchestrates classical time-series analysis tools through an LLM interface. It processes input pools containing thousands of real and synthetic univariate series stored in parquet files with explicit labels. The agent avoids direct raw data interpretation, instead planning budgeted calls to deterministic algorithms like anomaly detection and decomposition. A hard constraint ensures that all generated narratives are strictly grounded in the structured outputs provided by these tools. Consequently, the system produces a ranked list of potential anomalies accompanied by human-readable explanations. Each explanation details the nature, timing, and severity of detected issues while maintaining traceability to the underlying evidence. This approach allows for efficient triage of large datasets, directing human attention only to actionable events. The result is a cost-effective workflow that generates faithful, uncertainty-aware insights across various industrial and financial domains.

- **Input**: Pool of ~1000 – 4000 real univariate series (R1 + R2) and 1000+ synthetic univariate series stored as parquet files with columns `series_id`, `value`, `label` (point-wise anomaly flag)
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
| R1 | Combined IT (YAHOO, SMD, IOPS, Exathlon, WSD, NEK) | https://www.thedatum.org/datasets/TSB-UAD-Public-v2.zip | univariate series | point index / 1-minute / irregular (processed в TSB) | 1500 - 20000 | 500 - 1500 | The largest industrial IT-Ops pool. Web traffic, servers, KPIs, Spark logs, network flows |
| R2 | Combined Biomedical (MITDB, SVDB, LTDB) | https://www.thedatum.org/datasets/TSB-UAD-Public-v2.zip | univariate series | point index / irregular (processed в TSB) | 5000 - 30000 | 1000 - 3000 | Purely medical pool — ECG records (arrhythmias, long-term monitoring) |

- **Spare datasets**: SWaT, GECCO, CreditCard

### Synthetic generators (2)
| ID | Generator name | Base process | Anomaly families injected | Controls (rate/severity/duration) |
|----|----------------|-------------|----------------------------|----------------------------------|
| S1 | Stationary | white_noise, ar1, ar2 | point, group, level_shift, variance | rate=0.5, severity=2.6–4.0σ, duration=2.5–5% of series |
| S2 | Trend-Seasonal | linear_trend, seasonal_sine, trend_seasonal | trend, seasonality, group, level_shift | rate=0.5, severity=2.6–4.0σ, duration=2.5–5% of series |

## Unified Data Format

- **Canonical storage format**: Separate Parquet files per group (`R1.parquet`, `R2.parquet`, `S1.parquet`, `S2.parquet`). Each file contains the entire pool for that group. Columns (all series flattened):
  - `series_id` (string, primary key)
  - `time_index` (int64) – monotonic integer starting at 0 **per series**
  - `value` (float64)
  - `label` (int8) – point-wise anomaly flag (0/1) from TSB-UAD

- **Series ID**: `{group}__{original_dataset}__{original_id}_{sample_id}`
  - `group`: `R1` / `R2` / `S1` / `S2`
  - `original_dataset`: e.g. `YAHOO`, `MITDB`, `SMD`
  - `original_id`: original identifier from source (e.g. `real_42`, `record_117`)
  - `sample_id`:
    - `"full"` – если оригинальная серия **не разрезалась**
    - `"chunk{N}"` – где `N` = 0-based индекс чанка (пример: `R1__YAHOO__real_42_chunk0`)

  **Splitting rule (applied once during pool creation)**:
  - If the length of the original series **L ≤ 20,000** then set `sample_id = "full"`
  - If **L > 20,000** then create non-overlapping chunks of 20,000 points.
    - The final remainder is retained only if it contains ≥ 1,000 points.
    - Each point (and its `label`) is assigned to exactly one chunk.

- **Timestamps**: Absent. Only `time_index` (0, 1, 2, …) is used within each series. Missing values are handled according to TSB-UAD rules (forward-fill or drop).

- **Values**: `float64`, **raw** values from TSB-UAD (no additional scaling or unit conversion).

- **Metadata**: A separate small parquet file `metadata.parquet` (one per group) with the following columns:
  - `series_id`
  - `length`
  - `num_point_anomalies`
  - `y_i` (series-level label, see below)
  - `is_split` (boolean)
  - `original_length`
  - `source_notes` (optional)

## Label Definition (Series-Level)

### What is the prediction target?
- **Label type**: binary series-level label `y_i ∈ {0,1}`
- **Positive class definition**: anomalous time series (the series contains at least one anomalous point)
- **Negative class definition**: completely normal series
- **Ambiguous / unlabeled cases**: none (all series are assigned a `y_i`)

### How labels are derived (per dataset)
- **R1 derivation**: `y_i = 1` if the series (or chunk) contains at least one point with `label=1` (TSB-UAD point-wise labels). Otherwise, `y_i = 0`.
- **R2 derivation**: same rule applies (point-wise to series-level).
- **S1 derivation**: injected anomaly results in `y_i = 1`
- **S2 derivation**: injected anomaly results in `y_i = 1`

### Known label noise / caveats
- R1 and R2: slight subjective noise may be present. Synthetic data has zero noise.
- When splitting long series, an anomaly will fall into only one chunk — this is expected and is correctly reflected in the `y_i` of each chunk.  

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
