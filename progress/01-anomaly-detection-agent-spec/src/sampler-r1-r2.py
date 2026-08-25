"""Time-series sampler that turns the raw TSB-UAD pools into the R1/R2 pools of spec.md.

Reads the raw series of a group, splits long ones into period-aligned chunks, keeps the
share of anomalous points inside each sample within budget, and writes ``{group}.parquet``
plus ``{group}_metadata.parquet`` in the unified format.

Every tunable knob lives in the CONFIGURATION block below.

Guarantees
----------
Disjoint coverage
    Chunks of one series never overlap: an expansion is reported back through
    ``_end_idx`` and the chunk loop resumes from there, so a point belongs to one sample.

Unique keys
    ``series_id`` is unique. Whole-series strategies are decided once per series, before
    chunking starts, so they cannot race the per-chunk path.

Ratio budget
    Chunk samples satisfy ``anomaly_ratio <= ACCEPTABLE_ANOMALY_RATIO``; dilution aims for
    TARGET_ANOMALY_RATIO first. A series emitted undivided has no better dilution
    available and is bounded by MAX_ANOMALY_RATIO instead, tagged ``whole_series``.

Length budget
    Every sample satisfies ``ABSOLUTE_MIN <= length <= ABSOLUTE_MAX``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ======================================================================================
# CONFIGURATION - every knob for sampling lives in this block
# ======================================================================================

# raw_data sits next to the repository; pools go to <repo>/data/<task-name>/real/.
# Both resolve from this file, so the script behaves the same whatever directory it is
# launched from, and follows a rename of the task directory.
_HERE = Path(__file__).resolve()
RAW_DATA_DIR = _HERE.parents[4] / "raw_data"                                # PROJECT/raw_data
OUTPUT_DIR = _HERE.parents[3] / "data" / _HERE.parents[1].name / "real"     # tsagent/data/<task>/real

# Which pools to build, and the sub-directory of RAW_DATA_DIR each one reads.
GROUPS_TO_BUILD: Dict[str, str] = {"R1": "R1", "R2": "R2"}

# --- Chunking ---
SPLIT_THRESHOLD = 15000     # series longer than this are split into chunks
TARGET_CHUNK_MIN = 1500     # preferred chunk size; growth beyond it must be justified
TARGET_CHUNK_MAX = 8000     # soft cap on chunk size (a single period may exceed it)
ABSOLUTE_MIN = 1000         # hard lower bound on an emitted sample
ABSOLUTE_MAX = 35000        # hard upper bound on an emitted sample

# --- Anomaly budget (share of anomalous points inside one sample) ---
TARGET_ANOMALY_RATIO = 0.06        # dilution aims here first
ACCEPTABLE_ANOMALY_RATIO = 0.15    # fallback bound; every emitted sample satisfies it
MAX_ANOMALY_RATIO = 0.27           # above this a whole series is treated as degenerate

# --- Anomaly cluster handling ---
ANOMALY_LOOKAHEAD = 100     # gap tolerated when joining anomalies into one cluster

# --- Representativeness (clean samples only) ---
STATS_TOLERANCE = 0.65      # allowed relative deviation of chunk mean/std from the series

# --- Period detection ---
MIN_PERIOD_DETECT = 25
MAX_PERIOD_DETECT = 5000
MIN_PERIODS_PER_CHUNK = 3   # a chunk should span at least this many periods
MIN_LENGTH_FOR_PERIOD = 150 # below this, periodicity is not worth estimating

# ======================================================================================


@dataclass
class SampleRecord:
    """A single sampled time-series segment in the unified format."""

    series_id: str
    time_index: np.ndarray
    value: np.ndarray
    label: np.ndarray
    length: int
    num_point_anomalies: int
    y_i: int
    is_split: bool
    original_length: int
    source_notes: Optional[str] = None
    period_detected: Optional[int] = None
    is_representative: bool = True
    anomaly_ratio: float = 0.0
    # Absolute bounds inside the original series; the chunk loop resumes from _end_idx,
    # which is what keeps consecutive chunks disjoint.
    _start_idx: int = field(default=0, repr=False)
    _end_idx: int = field(default=0, repr=False)

    def to_dataframe(self) -> pd.DataFrame:
        """Flat frame for parquet storage."""
        return pd.DataFrame(
            {
                "series_id": self.series_id,
                "time_index": self.time_index,
                "value": self.value.astype(np.float64),
                "label": self.label.astype(np.int8),
            }
        )

    def to_metadata_row(self) -> Dict:
        """Metadata row for ``{group}_metadata.parquet``."""
        return {
            "series_id": self.series_id,
            "length": self.length,
            "num_point_anomalies": self.num_point_anomalies,
            "y_i": self.y_i,
            "is_split": self.is_split,
            "original_length": self.original_length,
            "source_notes": self.source_notes or "",
            "period_detected": self.period_detected,
            "is_representative": self.is_representative,
            "anomaly_ratio": round(self.anomaly_ratio, 4),
        }


class TimeSeriesSampler:
    """Adaptive sampler with period-aware chunking and an explicit anomaly budget."""

    def __init__(
        self,
        split_threshold: int = SPLIT_THRESHOLD,
        target_chunk_min: int = TARGET_CHUNK_MIN,
        target_chunk_max: int = TARGET_CHUNK_MAX,
        absolute_min: int = ABSOLUTE_MIN,
        absolute_max: int = ABSOLUTE_MAX,
        target_anomaly_ratio: float = TARGET_ANOMALY_RATIO,
        acceptable_anomaly_ratio: float = ACCEPTABLE_ANOMALY_RATIO,
        max_anomaly_ratio: float = MAX_ANOMALY_RATIO,
        anomaly_lookahead: int = ANOMALY_LOOKAHEAD,
        stats_tolerance: float = STATS_TOLERANCE,
        min_period_detect: int = MIN_PERIOD_DETECT,
        max_period_detect: int = MAX_PERIOD_DETECT,
    ) -> None:
        if not 0 < target_anomaly_ratio <= acceptable_anomaly_ratio <= max_anomaly_ratio < 1:
            raise ValueError(
                "anomaly ratios must satisfy 0 < target <= acceptable <= max < 1, got "
                f"{target_anomaly_ratio}, {acceptable_anomaly_ratio}, {max_anomaly_ratio}"
            )
        if not 0 < absolute_min <= target_chunk_min <= target_chunk_max <= absolute_max:
            raise ValueError(
                "chunk sizes must satisfy 0 < absolute_min <= target_min <= target_max <= absolute_max"
            )

        self.split_threshold = split_threshold
        self.target_chunk_min = target_chunk_min
        self.target_chunk_max = target_chunk_max
        self.absolute_min = absolute_min
        self.absolute_max = absolute_max
        self.target_anomaly_ratio = target_anomaly_ratio
        self.acceptable_anomaly_ratio = acceptable_anomaly_ratio
        self.max_anomaly_ratio = max_anomaly_ratio
        self.anomaly_lookahead = anomaly_lookahead
        self.stats_tolerance = stats_tolerance
        self.min_period_detect = min_period_detect
        self.max_period_detect = max_period_detect

        self.rejected_count = 0     # samples dropped for violating the anomaly budget
        self.degenerate_count = 0   # series salvaged for clean stretches only

    # ----------------------------------------------------------------------------------
    # Signal analysis
    # ----------------------------------------------------------------------------------

    def detect_period(self, values: np.ndarray) -> Optional[int]:
        """Dominant period via FFT autocorrelation, with detrending and a correlation check.

        The ceiling adapts to the series length; a fixed one would silently disable
        detection for every series shorter than three times that ceiling.
        """
        n = values.size
        max_p = min(self.max_period_detect, n // MIN_PERIODS_PER_CHUNK)
        if max_p < self.min_period_detect:
            return None

        v = values - np.linspace(values[0], values[-1], n)
        std = np.std(v)
        if std < 1e-10:
            return None
        v = (v - np.mean(v)) / std

        f = np.fft.rfft(v, n=2 * n)
        acf = np.fft.irfft(f * np.conjugate(f), n=2 * n)[:n]
        acf = acf / (acf[0] + 1e-10)

        threshold = max(0.2, 1.96 / np.sqrt(n))
        search_limit = min(max_p, n // 2 - 1)
        for lag in range(self.min_period_detect, max(self.min_period_detect + 1, search_limit)):
            if acf[lag] > acf[lag - 1] and acf[lag] > acf[lag + 1] and acf[lag] > threshold:
                # Confirm on the raw series, else detrending artefacts read as periods.
                with np.errstate(invalid="ignore"):
                    corr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
                if np.isfinite(corr) and corr >= 0.15:
                    return lag
        return None

    def compute_optimal_chunk_size(self, period: Optional[int]) -> int:
        """Chunk size: start from the preferred minimum, round up to whole periods."""
        size = self.target_chunk_min
        if period and period >= self.min_period_detect:
            size = max(size, period * MIN_PERIODS_PER_CHUNK)
            size = ((size + period - 1) // period) * period
        # The soft cap may be exceeded only to fit a single period.
        cap = max(self.target_chunk_max, period or 0)
        return int(np.clip(size, self.absolute_min, min(cap, self.absolute_max)))

    def align_boundary(self, pos: int, period: Optional[int], length: int, is_start: bool) -> int:
        """Snap a position to a period boundary: down for a start, up for an end."""
        if not period or period < self.min_period_detect:
            return pos
        if is_start:
            return max(0, (pos // period) * period)
        return min(length, ((pos + period - 1) // period) * period)

    def expand_anomaly_cluster(
        self, labels: np.ndarray, start: int, end: int, period: Optional[int] = None
    ) -> Tuple[int, int]:
        """Push ``end`` forward so a cluster of nearby anomalies is not cut in half.

        Capped at ``absolute_max`` from ``start``; an unbounded walk would swallow the
        rest of the series.
        """
        cap = min(labels.size, start + self.absolute_max)
        if end >= cap:
            return start, cap

        anomalies = np.flatnonzero(labels[start:end] == 1)
        if anomalies.size == 0:
            return start, end

        pos = start + int(anomalies[-1])
        new_end = end
        while True:
            window_start = pos + 1
            window_end = min(cap, window_start + self.anomaly_lookahead)
            if window_start >= window_end:
                break
            hits = np.flatnonzero(labels[window_start:window_end] == 1)
            if hits.size == 0:
                break
            pos = window_start + int(hits[-1])
            new_end = pos + 1

        new_end = self.align_boundary(new_end, period, labels.size, is_start=False)
        return start, min(new_end, cap)

    def check_representativeness(self, chunk: np.ndarray, g_mean: float, g_std: float) -> bool:
        """Whether a chunk's mean and spread are close enough to the whole series'.

        Both tolerances are in units of the series' spread. Scaling the mean tolerance by
        ``|g_mean|`` instead would collapse the band to nothing for a series centred near
        zero, rejecting almost every clean chunk of it.
        """
        scale = max(g_std, 1e-10)
        mean_ok = abs(float(np.mean(chunk)) - g_mean) <= self.stats_tolerance * scale
        std_ok = abs(float(np.std(chunk)) - g_std) <= self.stats_tolerance * scale
        return bool(mean_ok and std_ok)

    @staticmethod
    def _recalculate_label(labels: np.ndarray) -> Tuple[int, int, float]:
        """Series-level label, anomaly count and ratio from point-wise labels."""
        count = int(np.sum(labels))
        length = labels.size
        return (1 if count > 0 else 0), count, (count / length if length else 0.0)

    # ----------------------------------------------------------------------------------
    # Dilution
    # ----------------------------------------------------------------------------------

    def _optimize_expansion(
        self, labels: np.ndarray, start: int, end: int, total_len: int
    ) -> int:
        """Smallest forward expansion that brings the anomaly ratio down.

        Aims for ``target_anomaly_ratio``, falling back to ``acceptable_anomaly_ratio``
        only when the target is unreachable. Exact prefix-sum sweep: the ratio is not
        monotonic in ``end``, so a binary search would be unsound.
        """
        limit = min(total_len, start + self.absolute_max)
        if limit <= end:
            return end

        # csum[k] = anomalies in [start, start + k)  ->  ratio at length k is csum[k] / k
        csum = np.concatenate(([0], np.cumsum(labels[start:limit], dtype=np.int64)))
        lengths = np.arange(csum.size)
        ratios = np.divide(csum, lengths, out=np.full(csum.size, np.inf), where=lengths > 0)

        lo = (end - start) + 1  # first candidate length strictly beyond the current end
        if lo >= csum.size:
            return end

        for threshold in (self.target_anomaly_ratio, self.acceptable_anomaly_ratio):
            hits = np.flatnonzero(ratios[lo:] <= threshold)
            if hits.size:
                return start + lo + int(hits[0])
        return end

    # ----------------------------------------------------------------------------------
    # Sample construction
    # ----------------------------------------------------------------------------------

    def _finalize_sample(
        self,
        series_df: pd.DataFrame,
        group: str,
        dataset: str,
        orig_id: str,
        sample_id: str,
        start: int,
        end: int,
        is_split: bool,
        orig_len: int,
        g_stats: Dict,
        period: Optional[int],
        notes: str = "",
        ratio_cap: Optional[float] = None,
    ) -> Optional[SampleRecord]:
        """Validate length and representativeness, then build the record.

        The only place a :class:`SampleRecord` is created, so length bounds and label/ratio
        bookkeeping cannot diverge between code paths.
        """
        end = min(end, start + self.absolute_max, orig_len)
        if end - start < self.absolute_min:
            return None

        chunk = series_df.iloc[start:end]
        values = chunk["value"].to_numpy()
        labels = chunk["label"].to_numpy()
        y_i, anom_count, ratio = self._recalculate_label(labels)

        representative = True
        if y_i == 0:
            # Only clean samples are screened: an anomalous chunk is expected to deviate,
            # so the test would reject exactly what we want to keep.
            representative = self.check_representativeness(values, g_stats["mean"], g_stats["std"])
            if not representative:
                expanded_end = min(orig_len, start + self.absolute_max, end + min((end - start) // 5, 2000))
                if expanded_end > end:
                    candidate = series_df["value"].to_numpy()[start:expanded_end]
                    if self.check_representativeness(candidate, g_stats["mean"], g_stats["std"]):
                        end = expanded_end
                        chunk = series_df.iloc[start:end]
                        values = chunk["value"].to_numpy()
                        labels = chunk["label"].to_numpy()
                        y_i, anom_count, ratio = self._recalculate_label(labels)
                        representative = True
                if not representative:
                    self.rejected_count += 1
                    return None

        if ratio > (self.acceptable_anomaly_ratio if ratio_cap is None else ratio_cap):
            self.rejected_count += 1
            return None

        note_parts = [p for p in (f"period={period}" if period else "", notes) if p]
        return SampleRecord(
            series_id=f"{group}__{dataset}__{orig_id}_{sample_id}",
            time_index=np.arange(end - start, dtype=np.int64),
            value=values,
            label=labels,
            length=end - start,
            num_point_anomalies=anom_count,
            y_i=y_i,
            is_split=is_split,
            original_length=orig_len,
            source_notes=";".join(note_parts) or None,
            period_detected=period,
            is_representative=representative,
            anomaly_ratio=ratio,
            _start_idx=start,
            _end_idx=end,
        )

    def _create_chunk_sample(
        self,
        series_df: pd.DataFrame,
        group: str,
        dataset: str,
        orig_id: str,
        sample_id: str,
        start: int,
        end: int,
        is_split: bool,
        orig_len: int,
        g_stats: Dict,
        period: Optional[int],
    ) -> Optional[SampleRecord]:
        """Apply the local anomaly budget to one chunk, diluting it if necessary.

        Purely chunk-local; whole-series strategies are decided in :meth:`process_series`.
        """
        labels_full = series_df["label"].to_numpy()
        _, anom_count, ratio = self._recalculate_label(labels_full[start:end])

        if ratio > self.target_anomaly_ratio:
            expanded_end = self._optimize_expansion(labels_full, start, end, orig_len)
            if expanded_end > end:
                end = expanded_end

        return self._finalize_sample(
            series_df, group, dataset, orig_id, sample_id, start, end,
            is_split, orig_len, g_stats, period,
        )

    def _extract_clean_chunks(
        self,
        series_df: pd.DataFrame,
        group: str,
        dataset: str,
        orig_id: str,
        period: Optional[int],
        g_stats: Dict,
    ) -> List[SampleRecord]:
        """Salvage the anomaly-free stretches of a series that is mostly anomalous.

        Spans are clean by construction: alignment is applied only when it does not pull a
        labelled point into the span.
        """
        labels = series_df["label"].to_numpy()
        n = labels.size
        orig_len = n
        samples: List[SampleRecord] = []
        pos = 0

        while pos < n:
            if labels[pos] == 1:
                pos += 1
                continue

            limit = min(n, pos + self.absolute_max)
            run = np.flatnonzero(labels[pos:limit] == 1)
            end = pos + int(run[0]) if run.size else limit

            aligned = self.align_boundary(end, period, n, is_start=False)
            if aligned > end and aligned <= limit and not labels[end:aligned].any():
                end = aligned

            if end - pos >= self.absolute_min:
                record = self._finalize_sample(
                    series_df, group, dataset, orig_id, f"clean{len(samples)}",
                    pos, end, True, orig_len, g_stats, period, notes="clean_salvage",
                )
                if record is not None:
                    samples.append(record)
                    pos = record._end_idx
                    continue
            pos = max(end, pos + 1)

        return samples

    def _merge_tail(
        self, last: SampleRecord, values: np.ndarray, labels: np.ndarray, start: int, end: int
    ) -> Optional[SampleRecord]:
        """Append a too-short trailing span to the previous sample.

        Returns ``None`` if the merge would break the length or anomaly budget; the caller
        then drops the tail rather than emitting a sample that violates the invariants.
        """
        merged_value = np.concatenate([last.value, values[start:end]])
        if merged_value.size > self.absolute_max:
            return None

        merged_label = np.concatenate([last.label, labels[start:end]])
        y_i, anom_count, ratio = self._recalculate_label(merged_label)
        if ratio > self.acceptable_anomaly_ratio:
            return None

        return SampleRecord(
            series_id=last.series_id,
            time_index=np.arange(merged_value.size, dtype=np.int64),
            value=merged_value,
            label=merged_label,
            length=merged_value.size,
            num_point_anomalies=anom_count,
            y_i=y_i,
            is_split=True,
            original_length=last.original_length,
            source_notes=";".join(p for p in (last.source_notes, "tail_merged") if p),
            period_detected=last.period_detected,
            is_representative=last.is_representative,
            anomaly_ratio=ratio,
            _start_idx=last._start_idx,
            _end_idx=end,
        )

    # ----------------------------------------------------------------------------------
    # Series-level orchestration
    # ----------------------------------------------------------------------------------

    def process_series(
        self, series_df: pd.DataFrame, group: str, dataset: str, orig_id: str
    ) -> List[SampleRecord]:
        """Turn one raw series into zero or more samples."""
        values = series_df["value"].to_numpy()
        labels = series_df["label"].to_numpy()
        length = values.size
        if length == 0:
            return []

        g_stats = {"mean": float(np.mean(values)), "std": float(np.std(values))}
        period = self.detect_period(values) if length >= MIN_LENGTH_FOR_PERIOD else None
        global_ratio = float(np.sum(labels)) / length

        # --- Whole-series triage, decided exactly once ---------------------------------
        # Chunking cannot rescue a series whose global ratio already exceeds the budget.
        # Deciding this up front, not per chunk, keeps series_id unique and coverage
        # disjoint.
        if global_ratio > self.max_anomaly_ratio:
            self.degenerate_count += 1
            return self._extract_clean_chunks(series_df, group, dataset, orig_id, period, g_stats)

        if global_ratio > self.acceptable_anomaly_ratio and length <= self.absolute_max:
            record = self._finalize_sample(
                series_df, group, dataset, orig_id, "full", 0, length,
                False, length, g_stats, period, notes="whole_series",
                ratio_cap=self.max_anomaly_ratio,
            )
            return [record] if record else []

        # --- Short series: a single sample --------------------------------------------
        if length <= self.split_threshold:
            record = self._create_chunk_sample(
                series_df, group, dataset, orig_id, "full", 0, length,
                False, length, g_stats, period,
            )
            return [record] if record else []

        # --- Long series: disjoint chunks ---------------------------------------------
        return self._chunk_series(series_df, group, dataset, orig_id, period, g_stats)

    def _chunk_series(
        self,
        series_df: pd.DataFrame,
        group: str,
        dataset: str,
        orig_id: str,
        period: Optional[int],
        g_stats: Dict,
    ) -> List[SampleRecord]:
        """Split a long series into non-overlapping, budget-respecting chunks."""
        values = series_df["value"].to_numpy()
        labels = series_df["label"].to_numpy()
        length = values.size
        chunk_size = self.compute_optimal_chunk_size(period)

        samples: List[SampleRecord] = []
        pos = 0
        chunk_idx = 0

        while pos < length:
            start = pos
            end = min(pos + chunk_size, length)
            end = self.align_boundary(end, period, length, is_start=False)
            if np.any(labels[start:end] == 1):
                start, end = self.expand_anomaly_cluster(labels, start, end, period=period)

            # Too short to stand alone: merge into the previous sample, else drop.
            if end - start < self.absolute_min:
                if samples:
                    merged = self._merge_tail(samples[-1], values, labels, start, end)
                    if merged is not None:
                        samples[-1] = merged
                    else:
                        self.rejected_count += 1
                pos = max(end, start + 1)
                continue

            record = self._create_chunk_sample(
                series_df, group, dataset, orig_id, f"chunk{chunk_idx}", start, end,
                True, length, g_stats, period,
            )
            if record is not None:
                samples.append(record)
                chunk_idx += 1
                # Resume from the expanded boundary - this is what makes chunks disjoint.
                pos = max(record._end_idx, start + 1)
            else:
                pos = max(end, start + 1)

        return samples

    # ----------------------------------------------------------------------------------
    # Group-level orchestration
    # ----------------------------------------------------------------------------------

    @staticmethod
    def _load_series(path: Path) -> Optional[pd.DataFrame]:
        """Read one raw file and normalise it to a ``(value, label)`` frame."""
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        if df.empty:
            return None
        df.columns = [str(c).strip().lower() for c in df.columns]
        if "data" in df.columns and "value" not in df.columns:
            df = df.rename(columns={"data": "value"})
        if not {"value", "label"}.issubset(df.columns):
            return None

        df = df[["value", "label"]].copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce").astype("float64")
        df["label"] = (pd.to_numeric(df["label"], errors="coerce").fillna(0) > 0).astype("int8")
        # TSB-UAD rule: forward-fill gaps, drop any remaining at the head.
        df["value"] = df["value"].ffill()
        df = df[df["value"].notna()].reset_index(drop=True)
        return df if len(df) else None

    def process_group(self, raw_dir: Path, group: str, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process every dataset directory of a group and persist the pool."""
        self.rejected_count = 0
        self.degenerate_count = 0
        all_data: List[pd.DataFrame] = []
        all_meta: List[Dict] = []

        if not raw_dir.exists():
            logger.error("Raw directory not found: %s", raw_dir)
            return pd.DataFrame(), pd.DataFrame()

        dataset_dirs = sorted(d for d in raw_dir.iterdir() if d.is_dir())
        logger.info("=== Starting %s: found %d datasets ===", group, len(dataset_dirs))

        for ds_idx, ds_dir in enumerate(dataset_dirs, 1):
            ds_name = ds_dir.name
            ds_start = time.time()
            files = sorted(list(ds_dir.glob("*.csv")) + list(ds_dir.glob("*.parquet")))
            if not files:
                logger.warning("  [%d/%d] %s: no .csv/.parquet files", ds_idx, len(dataset_dirs), ds_name)
                continue
            logger.info("  [%d/%d] %s: %d files", ds_idx, len(dataset_dirs), ds_name, len(files))

            produced = 0
            for file_idx, fpath in enumerate(files, 1):
                try:
                    df = self._load_series(fpath)
                    if df is None:
                        continue
                    for sample in self.process_series(df, group, ds_name, fpath.stem):
                        all_data.append(sample.to_dataframe())
                        all_meta.append(sample.to_metadata_row())
                        produced += 1
                except Exception as exc:  # one broken file must not abort the pool
                    logger.error("    Error %s: %s", fpath.name, exc)
                if file_idx % 200 == 0:
                    logger.info("    ... %d/%d files", file_idx, len(files))

            logger.info("  [ok] %s: %d samples in %.1fs", ds_name, produced, time.time() - ds_start)

        if not all_data:
            logger.warning("No valid samples for %s", group)
            return pd.DataFrame(), pd.DataFrame()

        main_df = pd.concat(all_data, ignore_index=True)
        meta_df = pd.DataFrame(all_meta)
        main_df["time_index"] = main_df["time_index"].astype(np.int64)
        main_df["value"] = main_df["value"].astype(np.float64)
        main_df["label"] = main_df["label"].astype(np.int8)

        validate_pool(main_df, meta_df, self)

        output_dir.mkdir(parents=True, exist_ok=True)
        main_df.to_parquet(output_dir / f"{group}.parquet", index=False)
        meta_df.to_parquet(output_dir / f"{group}_metadata.parquet", index=False)

        self._log_summary(group, meta_df, output_dir)
        return main_df, meta_df

    def _log_summary(self, group: str, meta_df: pd.DataFrame, output_dir: Path) -> None:
        logger.info("=== %s COMPLETE ===", group)
        logger.info("Total samples: %d", len(meta_df))
        logger.info(
            "Length min/mean/max/std: %d / %.1f / %d / %.1f",
            meta_df["length"].min(), meta_df["length"].mean(),
            meta_df["length"].max(), meta_df["length"].std(),
        )
        logger.info(
            "Anomalous (y_i=1): %d (%.1f%%)", meta_df["y_i"].sum(), 100 * meta_df["y_i"].mean()
        )
        logger.info(
            "Anomaly ratio mean/max: %.4f / %.4f",
            meta_df["anomaly_ratio"].mean(), meta_df["anomaly_ratio"].max(),
        )
        logger.info("Period detected: %d (%.1f%%)",
                    meta_df["period_detected"].notna().sum(),
                    100 * meta_df["period_detected"].notna().mean())
        logger.info("Rejected samples (budget): %d", self.rejected_count)
        logger.info("Degenerate series (clean salvage): %d", self.degenerate_count)
        logger.info("Saved to %s/%s.parquet", output_dir, group)


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------

def validate_pool(data: pd.DataFrame, meta: pd.DataFrame, sampler: TimeSeriesSampler) -> None:
    """Assert the invariants promised in the module docstring, before anything is written."""
    if meta["series_id"].duplicated().any():
        dupes = meta.loc[meta["series_id"].duplicated(), "series_id"].tolist()[:5]
        raise AssertionError(f"duplicate series_id, e.g. {dupes}")
    if not np.isfinite(data["value"].to_numpy()).all():
        raise AssertionError("value contains NaN or inf")
    if not data["label"].isin((0, 1)).all():
        raise AssertionError("label must be binary")

    too_short = meta[meta["length"] < sampler.absolute_min]
    if len(too_short):
        raise AssertionError(f"{len(too_short)} samples shorter than absolute_min")
    too_long = meta[meta["length"] > sampler.absolute_max]
    if len(too_long):
        raise AssertionError(f"{len(too_long)} samples longer than absolute_max")

    # Chunk samples are held to the acceptable bound; whole-series samples to the max.
    whole = meta["source_notes"].fillna("").str.contains("whole_series")
    cap = np.where(whole, sampler.max_anomaly_ratio, sampler.acceptable_anomaly_ratio)
    over_budget = meta[meta["anomaly_ratio"] > cap + 1e-9]
    if len(over_budget):
        worst = over_budget["anomaly_ratio"].max()
        raise AssertionError(
            f"{len(over_budget)} samples exceed their anomaly-ratio budget (worst {worst:.4f})"
        )

    observed = data.groupby("series_id", sort=False).agg(
        obs_length=("time_index", "size"),
        obs_anomalies=("label", "sum"),
        last_index=("time_index", "max"),
    )
    joined = meta.set_index("series_id").join(observed, how="left")
    if joined["obs_length"].isna().any():
        raise AssertionError("metadata references a series_id absent from the data frame")
    if not (joined["obs_length"] == joined["length"]).all():
        raise AssertionError("metadata length disagrees with the number of rows")
    if not (joined["obs_anomalies"] == joined["num_point_anomalies"]).all():
        raise AssertionError("num_point_anomalies disagrees with the labels")
    if not (joined["y_i"] == (joined["obs_anomalies"] > 0).astype(int)).all():
        raise AssertionError("y_i disagrees with the point-wise labels")
    if not (joined["last_index"] == joined["length"] - 1).all():
        raise AssertionError("time_index must be contiguous and start at 0 for every series")


def main() -> None:
    """Build every pool listed in GROUPS_TO_BUILD."""
    sampler = TimeSeriesSampler()
    for group, subdir in GROUPS_TO_BUILD.items():
        logger.info("=== Processing %s ===", group)
        sampler.process_group(RAW_DATA_DIR / subdir, group, OUTPUT_DIR)


if __name__ == "__main__":
    main()
