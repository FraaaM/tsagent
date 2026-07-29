# Tasks

MVP scope (fixed for these 12 weeks): anomaly detection only; retrieve abnormal series from a pool of ~1,000 univariate industrial time series; full-series analysis; output a ranked list + recommended threshold (dynamic K); LLM tool-calling agent is mandatory and must plan budgeted deep-dives; evaluation on 2 real industrial datasets + 2 synthetic datasets.

| Week | Start date | End date | Status | Deliverable / acceptance criteria |
|------|------------|----------|--------|----------------------------------|
| 12: Full MVP sweep (2 real + 2 synthetic) + demo pack | 2026.05.26 | 2026.06.01 |  | Full matrix across all 4 datasets and budgets; deliver final curves/tables + 5-10 example flagged series with evidence-grounded rationales; confirm acceptance gates or document failure modes |
| 11: Comparative run + ablations (1 real + 1 synthetic) | 2026.05.19 | 2026.05.25 |  | Run agent vs baselines for at least 2 budgets; ablations: random deep-dive vs agent selection; fixed preset vs agent preset choice; report: AUPRC(B) + Recall@FPR(B) + cost |
| 10: LLM agent v0 (tool-calling + planning) | 2026.05.12 | 2026.05.18 |  | Agent selects B series for deep-dive using cheap summaries, chooses deep preset per series, produces final ranking + threshold policy; emits structured decision trace; guardrails: only cite tool outputs; only choose from enumerated presets/policies |
| 09: Baselines v0 | 2026.05.05 | 2026.05.11 |  | Baseline A: cheap-only; Baseline B: fixed two-stage (shortlist then deep-dive with fixed preset); Baseline C: random deep-dive; all output identical `ranked_list.json` schema |
| 08: Tool suite v0 (deep detect presets) | 2026.04.28 | 2026.05.04 |  | `deep_detect(series, preset)` implements a small preset menu (A/B/C) and returns a series-level anomaly score + diagnostics; deterministic replay works (same input/config => same output) |
| 07: Tool suite v0 (cheap scan) | 2026.04.21 | 2026.04.27 |  | `cheap_scan_all(pool)` runs on all 1k: robust cheap score + quality flags (missingness/flatline/volatility/spike proxy/step proxy); outputs structured JSON with stable IDs |
| 06: Budget + cost accounting | 2026.04.14 | 2026.04.20 |  | Standard run log: total tool calls, deep-dived count, per-tool runtime; quality-vs-budget curve support (vary B) |
| 05: Evaluation harness v0 (ranking + thresholding) | 2026.04.07 | 2026.04.13 |  | Given `ranked_list.json`, compute AUROC/AUPRC, PR curve, Recall@FPR; produce `metrics.json` and a single aggregated run summary |
| 04: Synthetic generator A + B v0 | 2026.03.31 | 2026.04.06 |  | Two synthetic generators produce 1k-series pools each with controlled anomaly rate + anomaly family tags; same unified format as real datasets |
| 03: Unified data format + pool builder v0 | 2026.03.24 | 2026.03.30 |  | `build_pool(dataset, seed)` creates a deterministic 1k-series pool with labels + metadata; export as a stable on-disk format (e.g., parquet/jsonl) |
| 02: Dataset commitment + label derivation | 2026.03.17 | 2026.03.23 |  | For each dataset: written label derivation to series-level, description of its statistical properties (standard EDA + trend, seasonality etc.) |
| 01: Anomaly detection agent spec | 2026.03.10 | 2026.03.16 | Doing | 2-4 page spec: exact label definition (series-level), budgets (B deep-dives per 1k), metrics (AUPRC/AUROC + Recall@FPR + cost curves), baseline definitions, acceptance gates; commit list of 2 real industrial datasets + 2 synthetic generators |

