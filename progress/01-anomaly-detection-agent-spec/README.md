# 01 — Anomaly Detection Agent Spec

Defines the task and builds the four evaluation pools it will run on. The full contract
lives in [`spec.md`](spec.md); this file covers what is done and how to rebuild the data.

## Status

- [x] Fill "TODO" fields left in `spec.md` — dataset-side sections are complete. The
      agent-side TODOs (tool suite, budget model, baseline tie-breaking, metrics artefacts)
      are deliberately still open; they belong to the next task.
- [x] Find 2 real datasets (about 1k time series) for anomaly detection — **R1** (IT-Ops)
      and **R2** (biomedical), sampled from TSB-UAD-Public-v2.
- [x] Build 2 synthetic datasets specifying the base processes and anomaly types —
      **S1** (stationary) and **S2** (trend-seasonal).

## Pools

| Pool | Source | Series | Mean length | `y_i = 1` | Max anomaly ratio |
|------|--------|--------|-------------|-----------|-------------------|
| R1 | YAHOO, SMD, IOPS, Exathlon, WSD, NEK | 3,828 | 2,503 | 42.7% | 0.241 |
| R2 | MITDB, SVDB, LTDB | 8,114 | 2,483 | 20.3% | 0.150 |
| S1 | `white_noise`, `ar1`, `ar2` | 1,000 | 2,525 | 45.3% | 0.050 |
| S2 | `linear_trend`, `seasonal_sine`, `trend_seasonal` | 1,000 | 2,518 | 51.1% | 0.050 |

13,942 series in total. Each pool is a pair of parquet files —
`{pool}.parquet` (`series_id`, `time_index`, `value`, `label`) and
`{pool}_metadata.parquet` (one row per series, `y_i` among the columns) — written under
`data/01-anomaly-detection-agent-spec/` at the repository root:

```
data/01-anomaly-detection-agent-spec/
  real/       R1.parquet, R1_metadata.parquet, R2.parquet, R2_metadata.parquet
  synthetic/  S1.parquet, S1_metadata.parquet, S2.parquet, S2_metadata.parquet,
              S1_manifest.json, S2_manifest.json
```

## Layout

```
spec.md                          task contract, data format, label definition
src/sampler-r1-r2.py             builds R1 + R2 from raw_data/
src/generator-s1-s2.py           builds S1 + S2 from scratch
src/raw_real_datasets_analysis.ipynb   exploration of the raw TSB-UAD files
src/real_datasets_analysis.ipynb       checks on the built R1/R2 pools
src/synthetic_datasets_analysis.ipynb  checks on the built S1/S2 pools
(pools land in ../../data/01-anomaly-detection-agent-spec/, git-ignored)
```

## Rebuilding the data

Both scripts are configuration-driven: every knob sits in the `CONFIGURATION` block at the
top of the file, and there are no command-line arguments. Edit the block, run the file.

Install the dependencies once:

```bash
pip install -r ../../code/requirements.txt
```

Real pools — expects `raw_data/R1/<dataset>/*.csv|parquet` next to the repository, i.e.
`PROJECT/raw_data/` (set by `RAW_DATA_DIR`). Takes a few minutes:

```bash
python src/sampler-r1-r2.py
```

Synthetic pools — no input data needed, runs in a few seconds:

```bash
python src/generator-s1-s2.py
```

The sampler writes into `data/<task>/real/`, the generator into `data/<task>/synthetic/`.
Both validate the result before writing: unique `series_id`,
contiguous `time_index`, metadata agreeing with the labels, and the length and
anomaly-ratio budgets. A violated invariant aborts the run instead of producing a file
later stages would trust.

`generator-s1-s2.py` also writes `{pool}_manifest.json` with the exact settings used, and
is reproducible to the byte: a series depends only on `(MASTER_SEED, pool, index)`, not on
pool size or generation order.

## Design decisions worth knowing

**Anomaly-ratio budget.** A positive series should *contain* an anomaly, not consist of
one. Chunks are diluted forward into clean data toward 0.06, accepted up to 0.15, and
dropped above it. A series whose global ratio exceeds 0.27 is unrescuable by any chunking,
so only its anomaly-free stretches are kept (`clean{N}`).

**Disjoint coverage.** Chunks of one original series never overlap. Points may be dropped
when a span busts the budget, but never duplicated — so a series-level metric cannot be
inflated by the same data appearing twice.

**Severity in local sigma.** Synthetic anomaly magnitudes are measured in
`MAD(diff(x))·1.4826/√2`, computed on the clean series. That statistic is invariant to
trend and seasonality, so severity 3.0 is equally visible on `white_noise` and on
`trend_seasonal`; the marginal standard deviation would be dominated by the trend.

**Honest labels.** Every synthetic perturbation is tapered to exactly zero at both segment
boundaries, so the labelled span covers every modified sample and a segment anomaly never
leaves an unlabelled discontinuity just outside its own label.
