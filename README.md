# tsagent

Code & notes on the Time Series Agent project

## Project description

**Topic**: Evidence-Grounded LLM Agent-based Orchestration of Time Series Analysis Tools for for Time Series Insights (Оркестрация алгоритмов анализа временных на основе LLM-агента)

**Description**: Build an offline agent that helps users triage large pools of univariate time series by orchestrating classical time-series tools (anomaly detection, change points, decomposition, forecasting) and producing evidence-linked, uncertainty-aware natural-language insights. Industrial monitoring is a primary source of inspiration, but the approach should transfer to other domains (IoT, finance, web metrics, medicine).

## Research objective

### Abstract

Many domains (industrial monitoring, IoT, finance, web metrics, medicine) involve large pools of time series where the key challenge is not running a single detector, but efficiently finding and explaining the few actionable events in the data. This project studies an LLM-based offline analysis agent that does not "interpret raw time series" directly; instead it plans and calls deterministic time-series analysis tools (decomposition, anomaly detection, change point detection, forecasting/backtesting, similarity search) and turns their structured outputs into concise reports with explicit evidence links. The focus is on faithfulness (no unsupported claims), robustness to missingness/resampling artifacts, and cost-aware exploration over many series.

### Key research question

Can a tool-augmented LLM agent produce accurate and faithful time-series narratives (what happened, when, how severe, and what to check next) while using a limited tool-call budget, and outperform fixed non-agent pipelines in usefulness and triage efficiency across multiple domains?

### Why this deserves studying

- Many real systems have large and heterogeneous collections of time series: one-size-fits-all detectors generate too many alerts, and manual investigation does not scale.
- LLMs are strong at planning and explanation, but are prone to hallucination; an evidence-grounded tool interface can make them reliable in data analysis workflows.
- A principled evaluation of "insight quality" (claim-to-evidence traceability + correctness + cost) is still underdeveloped compared to standard forecasting benchmarks.
- The resulting system is broadly usable: it can generate weekly/incident reports and help prioritize which series/entities deserve human attention in offline analysis.
