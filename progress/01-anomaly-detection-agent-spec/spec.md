# Anomaly Detection Agent Spec

## One-Paragraph Spec
The MVP functions as an offline agent that orchestrates classical time-series analysis tools through an LLM interface. It processes input pools containing thousands of real and synthetic univariate series stored in parquet files with explicit labels. The agent avoids direct raw data interpretation, instead planning budgeted calls to deterministic algorithms like anomaly detection and decomposition. A hard constraint ensures that all generated narratives are strictly grounded in the structured outputs provided by these tools. Consequently, the system produces a ranked list of potential anomalies accompanied by human-readable explanations. Each explanation details the nature, timing, and severity of detected issues while maintaining traceability to the underlying evidence. This approach allows for efficient triage of large datasets, directing human attention only to actionable events. The result is a cost-effective workflow that generates faithful, uncertainty-aware insights across various industrial and financial domains.

- **Input**: Pool of 11,942 real univariate series (R1 + R2) and 2,000 synthetic univariate series (S1 + S2) stored as parquet files with columns `series_id`, `time_index`, `value`, `label` (point-wise anomaly flag)
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
| R1 | Combined IT (YAHOO, SMD, IOPS, Exathlon, WSD, NEK) | https://www.thedatum.org/datasets/TSB-UAD-Public-v2.zip | univariate series | point index / 1-minute / irregular (processed in TSB) | 1,004 - 33,067 (mean 2,503) | 3,828 built | The largest industrial IT-Ops pool. Web traffic, servers, KPIs, Spark logs, network flows |
| R2 | Combined Biomedical (MITDB, SVDB, LTDB) | https://www.thedatum.org/datasets/TSB-UAD-Public-v2.zip | univariate series | point index / irregular (processed in TSB) | 1,046 - 34,607 (mean 2,483) | 8,114 built | Purely medical pool — ECG records (arrhythmias, long-term monitoring) |

- **Spare datasets**: SWaT, GECCO, CreditCard

### Synthetic generators (2)
| ID | Generator name | Base process | Anomaly families injected | Controls (rate/severity/duration) |
|----|----------------|-------------|----------------------------|----------------------------------|
| S1 | Stationary | white_noise, ar1, ar2 | point, group, level_shift, variance | 1,000 series, length 1,000–4,000, rate=0.5, severity=2.6–4.0σ, duration=2.5–5% of series |
| S2 | Trend-Seasonal | linear_trend, seasonal_sine, trend_seasonal | trend, seasonality, group, level_shift | 1,000 series, length 1,000–4,000, rate=0.5, severity=2.6–4.0σ, duration=2.5–5% of series |

- **Severity unit**: σ is `sigma_local`, a robust estimate of the innovation scale computed as `MAD(diff(x)) · 1.4826 / √2` on the **clean** series before injection. Differencing removes trend and attenuates seasonality, so a severity of 3.0 is equally visible on `white_noise` and on `trend_seasonal`. The marginal standard deviation is **not** used: on a trending series it is dominated by the trend and would make severity meaningless.
- **Base-process parameters** are drawn per series (AR coefficients from the stationarity triangle, seasonal period 12–200 with a second harmonic, trend parameterised by total rise) and recorded in `base_params` in the metadata.
- **Anomaly family definitions** — the six mechanisms are deliberately distinct so that no two collapse into the same signal:

| Family | Mechanism | Distinguishing property |
|--------|-----------|-------------------------|
| `point` | isolated global outliers of ±severity·σ | minimum spacing of 5 points is enforced, so spikes stay isolated |
| `group` | an alien deterministic waveform (square / triangle / sawtooth) is laid over the segment | the *shape* stops belonging to the generating process |
| `level_shift` | sustained mean step of severity·σ | sharp 2-sample edges, mean displaced, waveform preserved |
| `variance` | local dispersion inflated by (1 + severity/2), noise resampled | mean exactly preserved, waveform destroyed |
| `trend` | transient linear drift peaking at severity·σ | returns to zero by the segment end |
| `seasonality` | the seasonal period is warped by up to ±severity·12% | genuine frequency change, not an extra sine on top |

- **Labelling guarantee**: every perturbation is multiplied by a trapezoidal window that is exactly zero at both segment boundaries. The perturbed support therefore equals the labelled support — a segment anomaly never leaves an unlabelled discontinuity just outside its own label.
- **Reproducibility**: one independent RNG per series, derived from `MASTER_SEED` via `SeedSequence.spawn`. A series depends only on `(MASTER_SEED, group, series_index)`, so pools are unchanged by pool size, generation order or parallelism. Each run writes `{group}_manifest.json` with the exact settings used.

## Unified Data Format

- **Canonical storage format**: Separate Parquet files per group, written to `data/01-anomaly-detection-agent-spec/` at the repository root — `real/` holds `R1.parquet` and `R2.parquet`, `synthetic/` holds `S1.parquet` and `S2.parquet`, each beside its `*_metadata.parquet`. Each file contains the entire pool for that group. Columns (all series flattened):
  - `series_id` (string, primary key)
  - `time_index` (int64) – monotonic integer starting at 0 **per series**
  - `value` (float64)
  - `label` (int8) – point-wise anomaly flag (0/1) from TSB-UAD

- **Series ID**: `{group}__{original_dataset}__{original_id}_{sample_id}`
  - `group`: `R1` / `R2` / `S1` / `S2`
  - `original_dataset`: e.g. `YAHOO`, `MITDB`, `SMD`
  - `original_id`: original identifier from source (e.g. `real_42`, `record_117`)
  - `sample_id`:
    - `"full"` - the original series was **not split**
    - `"chunk{N}"` - `N` is the 0-based chunk index (example: `R1__YAHOO__real_42_chunk0`)
    - `"clean{N}"` - `N`-th anomaly-free stretch salvaged from a series whose global anomaly ratio exceeded `MAX_ANOMALY_RATIO` (such a series cannot be diluted into a usable sample, so only its clean spans are kept)

  For synthetic pools `original_dataset` is the literal `SYNTH` and `original_id` is `{base_type}-{index:05d}`, e.g. `S1__SYNTH__ar1-00042_full`, so the three-part `__` structure is identical to R1/R2.

  **Splitting rule (applied once during pool creation, see `src/sampler-r1-r2.py`)**:
  - If the length of the original series **L ≤ 15,000** (`SPLIT_THRESHOLD`) then set `sample_id = "full"`.
  - If **L > 15,000** then cut non-overlapping chunks. The chunk size starts at 1,500 points (`TARGET_CHUNK_MIN`) and, when a dominant period is detected, is rounded up to a whole number of periods spanning at least 3 of them, capped at 8,000 (`TARGET_CHUNK_MAX`).
  - A chunk boundary is snapped to a period boundary when a period is detected, and pushed forward when it would otherwise cut an anomaly cluster in half (gaps up to 100 points are treated as one cluster).
  - A trailing span shorter than 1,000 points is merged into the previous chunk; if that merge would break the length or anomaly budget, the span is dropped instead.
  - Every emitted sample satisfies **1,000 ≤ length ≤ 35,000** (`ABSOLUTE_MIN` / `ABSOLUTE_MAX`).
  - **Each point (and its `label`) is assigned to at most one sample** — chunks of one original series never overlap. Points may be dropped (a span that violates the anomaly budget is discarded), but never duplicated.

  **Anomaly-ratio budget**: a positive series should *contain* an anomaly, not consist of one. The share of anomalous points inside a sample is therefore bounded:
  - Dilution (extending a chunk forward into clean data) targets `TARGET_ANOMALY_RATIO = 0.06` and falls back to `ACCEPTABLE_ANOMALY_RATIO = 0.15` only when the target is unreachable within `ABSOLUTE_MAX`.
  - A chunk still above 0.15 after dilution is dropped.
  - A series whose **global** ratio exceeds `MAX_ANOMALY_RATIO = 0.27` cannot be rescued by any chunking, so only its anomaly-free stretches are kept (`clean{N}`).
  - A series whose global ratio lies in (0.15, 0.27] and which fits in `ABSOLUTE_MAX` is emitted undivided — that is the best dilution available — and is tagged `whole_series` in `source_notes`. These are the only samples allowed above 0.15.
  - Realised maxima: R1 = 0.241, R2 = 0.150, S1 = S2 = 0.050.

- **Timestamps**: Absent. Only `time_index` (0, 1, 2, …) is used within each series. Missing values are handled according to TSB-UAD rules (forward-fill or drop).

- **Values**: `float64`, **raw** values from TSB-UAD (no additional scaling or unit conversion).

- **Metadata**: A separate small parquet file `{group}_metadata.parquet` (one per group) with the following columns:
  - `series_id`
  - `length`
  - `num_point_anomalies`
  - `y_i` (series-level label, see below)
  - `is_split` (boolean)
  - `original_length`
  - `source_notes` (free-form provenance tags: `period=…`, `whole_series`, `clean_salvage`, `tail_merged`)

  Real pools (R1, R2) additionally carry:
  - `period_detected` (int or null) — dominant period, or null when none was found
  - `is_representative` (boolean) — whether a clean sample's mean/std stay within `STATS_TOLERANCE`·σ of the parent series
  - `anomaly_ratio` (float) — `num_point_anomalies / length`

  Synthetic pools (S1, S2) additionally carry:
  - `base_type`, `base_params` (JSON) — the generating process and its drawn parameters
  - `anomaly_type`, `severity`, `num_segments`, `segments` (JSON list of `{kind, start, end, severity, …}`)
  - `anomaly_fraction` (realised) and `target_fraction` (requested)

  The `segments` field gives exact ground-truth spans for the synthetic pools, which makes them usable for point-level evaluation as well as the series-level task defined below.

- **Integrity**: both scripts validate their output before writing it — unique `series_id`, contiguous `time_index` starting at 0, metadata agreeing with the actual labels, and the length and anomaly-ratio budgets. A violation aborts the run rather than producing a parquet file that later stages would trust.

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

### Realised class balance
| Pool | Series | Mean length | `y_i = 1` |
|------|--------|-------------|-----------|
| R1 | 3,828 | 2,503 | 42.7% |
| R2 | 8,114 | 2,483 | 20.3% |
| S1 | 1,000 | 2,525 | 45.3% |
| S2 | 1,000 | 2,518 | 51.1% |

### Known label noise / caveats
- R1 and R2: slight subjective noise may be present. Synthetic data has zero noise.
- When splitting long series, an anomaly will fall into only one chunk — this is expected and is correctly reflected in the `y_i` of each chunk.
- Anomaly-free spans salvaged from a degenerate series (`clean{N}`) are `y_i = 0` by construction: alignment to a period boundary is applied only when it does not pull a labelled point into the span.
- Dropping over-budget spans removes anomalous material preferentially, so the realised positive rate is lower than the raw share of dirty series in the source. This is intentional: it keeps positives interpretable as "contains an anomaly".

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
