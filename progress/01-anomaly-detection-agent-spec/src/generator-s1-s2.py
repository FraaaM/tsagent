"""Synthetic univariate time-series generator for the S1/S2 anomaly-detection pools.

Every tunable knob lives in the CONFIGURATION block below.

Guarantees
----------
Reproducibility
    One independent child RNG per series, so a series depends only on
    ``(MASTER_SEED, group, series_index)`` - not on pool size, pool order or parallelism.
    Each run writes a ``*_manifest.json`` with the exact settings used.

Severity scale
    Magnitudes are in units of ``sigma_local`` (MAD of first differences), which is
    invariant to trend and seasonality - so ``severity`` means the same thing on every
    base process.

Labelling
    Perturbations are tapered to exactly zero at both segment boundaries, so the perturbed
    support equals the labelled support and no unlabelled discontinuity is left behind.

Anomaly taxonomy
    ``point`` isolated outliers | ``group`` alien shapelet | ``level_shift`` mean step |
    ``variance`` dispersion inflation | ``trend`` transient drift | ``seasonality`` period
    warp - six mechanically distinct families.
"""

from __future__ import annotations

import json
import logging
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.signal import lfilter

__version__ = "2.0.0"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ======================================================================================
# CONFIGURATION - every knob for synthetic generation lives in this block
# ======================================================================================

# Destination for S1/S2, their metadata and the run manifests:
# <repo>/data/<task-name>/synthetic/. Derived from this file, so it follows a rename.
_HERE = Path(__file__).resolve()
OUTPUT_DIR = _HERE.parents[3] / "data" / _HERE.parents[1].name / "synthetic"

# Which pools to build on this run, e.g. ("S1",) to rebuild only the stationary pool.
GROUPS_TO_BUILD: tuple[str, ...] = ("S1", "S2")

MASTER_SEED = 42                        # root of all randomness; fixes the pools exactly
NUM_SERIES_PER_POOL = 1000              # series per pool
LENGTH_RANGE = (1000, 4000)             # per-series length, drawn uniformly
ANOMALY_RATE = 0.5                      # share of series that receive an anomaly
ANOMALY_FRACTION_RANGE = (0.025, 0.05)  # share of anomalous points inside a dirty series
SEVERITY_RANGE = (2.6, 4.0)             # anomaly magnitude in units of sigma_local

# Names must exist in BASE_PROCESSES / ANOMALY_INJECTORS; validated before generation.
POOL_DEFINITIONS: dict[str, dict[str, tuple[str, ...]]] = {
    "S1": {  # Stationary
        "base_types": ("white_noise", "ar1", "ar2"),
        "allowed_anomaly_types": ("point", "group", "level_shift", "variance"),
    },
    "S2": {  # Trend-Seasonal
        "base_types": ("linear_trend", "seasonal_sine", "trend_seasonal"),
        "allowed_anomaly_types": ("trend", "seasonality", "group", "level_shift"),
    },
}

# --- Structural constants; change only if the generation model itself changes. ---
MIN_SERIES_LENGTH = 64      # below this a tapered segment anomaly cannot be placed
AR_BURN_IN = 500            # samples dropped from an AR draw to remove the transient
POINT_MIN_SPACING = 5       # minimum gap between point outliers so they stay isolated
MAD_TO_SIGMA = 1.4826

# ======================================================================================


@dataclass(frozen=True)
class PoolConfig:
    """Declarative description of one synthetic pool."""

    group: str
    num_series: int
    base_types: Sequence[str]
    allowed_anomaly_types: Sequence[str]
    anomaly_rate: float
    length_range: tuple[int, int]
    anomaly_fraction_range: tuple[float, float]
    severity_range: tuple[float, float]
    seed: int

    def validate(self) -> None:
        """Fail fast on a configuration that cannot produce a valid pool."""
        if self.num_series <= 0:
            raise ValueError(f"num_series must be positive, got {self.num_series}")
        if not 0.0 <= self.anomaly_rate <= 1.0:
            raise ValueError(f"anomaly_rate must lie in [0, 1], got {self.anomaly_rate}")

        lo, hi = self.length_range
        if not MIN_SERIES_LENGTH <= lo <= hi:
            raise ValueError(
                f"length_range must satisfy {MIN_SERIES_LENGTH} <= lo <= hi, got {self.length_range}"
            )

        f_lo, f_hi = self.anomaly_fraction_range
        if not 0.0 < f_lo <= f_hi < 1.0:
            raise ValueError(
                f"anomaly_fraction_range must satisfy 0 < lo <= hi < 1, got {self.anomaly_fraction_range}"
            )

        s_lo, s_hi = self.severity_range
        if not 0.0 < s_lo <= s_hi:
            raise ValueError(f"severity_range must satisfy 0 < lo <= hi, got {self.severity_range}")

        if not self.base_types:
            raise ValueError("base_types must not be empty")
        unknown = sorted(set(self.base_types) - set(BASE_PROCESSES))
        if unknown:
            raise ValueError(f"unknown base_types: {unknown}")

        if self.anomaly_rate > 0 and not self.allowed_anomaly_types:
            raise ValueError("allowed_anomaly_types must not be empty when anomaly_rate > 0")
        unknown = sorted(set(self.allowed_anomaly_types) - set(ANOMALY_INJECTORS))
        if unknown:
            raise ValueError(f"unknown allowed_anomaly_types: {unknown}")

    def to_manifest(self) -> dict:
        """Serialise the configuration for the run manifest."""
        return {
            "generator_version": __version__,
            "group": self.group,
            "num_series": self.num_series,
            "base_types": list(self.base_types),
            "allowed_anomaly_types": list(self.allowed_anomaly_types),
            "anomaly_rate": self.anomaly_rate,
            "length_range": list(self.length_range),
            "anomaly_fraction_range": list(self.anomaly_fraction_range),
            "severity_range": list(self.severity_range),
            "seed": self.seed,
        }


def pool_config(group: str) -> PoolConfig:
    """Build the :class:`PoolConfig` for ``group`` from the CONFIGURATION block."""
    if group not in POOL_DEFINITIONS:
        raise KeyError(f"unknown pool {group!r}; known pools: {sorted(POOL_DEFINITIONS)}")
    definition = POOL_DEFINITIONS[group]
    return PoolConfig(
        group=group,
        num_series=NUM_SERIES_PER_POOL,
        base_types=definition["base_types"],
        allowed_anomaly_types=definition["allowed_anomaly_types"],
        anomaly_rate=ANOMALY_RATE,
        length_range=LENGTH_RANGE,
        anomaly_fraction_range=ANOMALY_FRACTION_RANGE,
        severity_range=SEVERITY_RANGE,
        seed=MASTER_SEED,
    )


@dataclass
class AnomalySegment:
    """A single labelled anomalous span, half-open ``[start, end)``."""

    kind: str
    start: int
    end: int
    severity: float
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "start": int(self.start),
            "end": int(self.end),
            "severity": round(float(self.severity), 4),
            **{k: _jsonable(v) for k, v in self.detail.items()},
        }


@dataclass
class SeriesResult:
    """One generated series together with its provenance."""

    series_id: str
    values: np.ndarray
    labels: np.ndarray
    metadata: dict


def _jsonable(value):
    """Coerce numpy scalars to plain Python types for JSON serialisation."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 6)
    return value


# --------------------------------------------------------------------------------------
# Base processes - each returns ``(values, params)``; params is recorded in the metadata.
# --------------------------------------------------------------------------------------

def _base_white_noise(rng: np.random.Generator, length: int) -> tuple[np.ndarray, dict]:
    sigma = float(rng.uniform(0.5, 1.5))
    return rng.normal(0.0, sigma, length), {"sigma": sigma}


def _sample_ar1_phi(rng: np.random.Generator) -> float:
    """Draw a stationary AR(1) coefficient, avoiding the near-unit-root regime."""
    sign = -1.0 if rng.random() < 0.2 else 1.0
    return float(rng.uniform(0.3, 0.9)) * sign


def _sample_ar2_phi(rng: np.random.Generator) -> tuple[float, float]:
    """Rejection-sample AR(2) coefficients from the stationarity triangle.

    The margin keeps roots off the unit circle; without it a pool advertised as stationary
    would contain near-random-walk draws.
    """
    margin = 0.05
    for _ in range(100):
        phi1 = float(rng.uniform(-1.6, 1.6))
        phi2 = float(rng.uniform(-0.9, 0.6))
        if abs(phi2) < 1.0 - margin and phi1 + phi2 < 1.0 - margin and phi2 - phi1 < 1.0 - margin:
            return phi1, phi2
    return 0.6, -0.3  # deterministic stationary fallback


def _ar_filter(rng: np.random.Generator, length: int, phis: Sequence[float], sigma: float) -> np.ndarray:
    """Draw an AR(p) realisation with the burn-in transient removed.

    ``lfilter`` starts from zero initial conditions, so the leading samples are
    under-dispersed - which would read as a variance anomaly at the head of every series.
    """
    innovations = rng.normal(0.0, sigma, length + AR_BURN_IN)
    denominator = np.concatenate(([1.0], -np.asarray(phis, dtype=np.float64)))
    return lfilter([1.0], denominator, innovations)[AR_BURN_IN:]


def _base_ar1(rng: np.random.Generator, length: int) -> tuple[np.ndarray, dict]:
    phi = _sample_ar1_phi(rng)
    sigma = float(rng.uniform(0.4, 1.0))
    return _ar_filter(rng, length, [phi], sigma), {"phi": phi, "sigma": sigma}


def _base_ar2(rng: np.random.Generator, length: int) -> tuple[np.ndarray, dict]:
    phi1, phi2 = _sample_ar2_phi(rng)
    sigma = float(rng.uniform(0.4, 1.0))
    return _ar_filter(rng, length, [phi1, phi2], sigma), {"phi1": phi1, "phi2": phi2, "sigma": sigma}


def _base_linear_trend(rng: np.random.Generator, length: int) -> tuple[np.ndarray, dict]:
    sigma = float(rng.uniform(0.3, 0.8))
    # Parameterised by total rise, so the slope stays meaningful at any length.
    total_rise = float(rng.uniform(4.0, 25.0)) * (1.0 if rng.random() < 0.5 else -1.0)
    slope = total_rise / length
    t = np.arange(length, dtype=np.float64)
    return slope * t + rng.normal(0.0, sigma, length), {
        "slope": slope,
        "total_rise": total_rise,
        "sigma": sigma,
    }


def _draw_period(rng: np.random.Generator, length: int) -> int:
    """Draw a seasonal period that fits several times into the series."""
    max_period = max(12, min(200, length // 8))
    return int(rng.integers(12, max_period + 1))


def _base_seasonal_sine(rng: np.random.Generator, length: int) -> tuple[np.ndarray, dict]:
    period = _draw_period(rng, length)
    amp = float(rng.uniform(1.0, 3.5))
    sigma = float(rng.uniform(0.2, 0.7))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    # A second harmonic keeps the waveform from being a textbook sine.
    harmonic = float(rng.uniform(0.0, 0.35))
    t = np.arange(length, dtype=np.float64)
    values = (
        amp * np.sin(2.0 * np.pi * t / period + phase)
        + amp * harmonic * np.sin(4.0 * np.pi * t / period + phase)
        + rng.normal(0.0, sigma, length)
    )
    return values, {
        "period": period,
        "amp": amp,
        "phase": phase,
        "harmonic": harmonic,
        "sigma": sigma,
    }


def _base_trend_seasonal(rng: np.random.Generator, length: int) -> tuple[np.ndarray, dict]:
    seasonal, params = _base_seasonal_sine(rng, length)
    total_rise = float(rng.uniform(4.0, 20.0)) * (1.0 if rng.random() < 0.5 else -1.0)
    slope = total_rise / length
    t = np.arange(length, dtype=np.float64)
    return seasonal + slope * t, {**params, "slope": slope, "total_rise": total_rise}


BASE_PROCESSES: dict[str, Callable[[np.random.Generator, int], tuple[np.ndarray, dict]]] = {
    "white_noise": _base_white_noise,
    "ar1": _base_ar1,
    "ar2": _base_ar2,
    "linear_trend": _base_linear_trend,
    "seasonal_sine": _base_seasonal_sine,
    "trend_seasonal": _base_trend_seasonal,
}


# --------------------------------------------------------------------------------------
# Scale estimation and windowing
# --------------------------------------------------------------------------------------

def local_scale(values: np.ndarray) -> float:
    """Robust innovation scale: ``MAD(diff(x)) * 1.4826 / sqrt(2)``.

    Differencing removes trend and attenuates seasonality; the MAD resists existing
    structure. This is the unit ``severity`` is measured in.
    """
    if values.size < 2:
        return 1.0
    d = np.diff(values)
    mad = float(np.median(np.abs(d - np.median(d))))
    scale = MAD_TO_SIGMA * mad / np.sqrt(2.0)
    if scale > 1e-9:
        return scale
    # Degenerate (near-constant) series: fall back to the marginal spread.
    fallback = float(np.std(values))
    return fallback if fallback > 1e-9 else 1.0


def taper(duration: int, ramp: int) -> np.ndarray:
    """Trapezoidal window, exactly zero at both ends.

    Keeps the series continuous at the segment boundaries, so the labelled span covers
    every modified sample and nothing outside it is disturbed.
    """
    ramp = int(np.clip(ramp, 1, max(1, duration // 2)))
    w = np.ones(duration, dtype=np.float64)
    edge = np.linspace(0.0, 1.0, ramp, endpoint=False) if ramp > 1 else np.zeros(1)
    w[:ramp] = edge
    w[duration - ramp:] = edge[::-1]
    return w


def _draw_segment(rng: np.random.Generator, length: int, duration: int) -> tuple[int, int]:
    """Place a segment of ``duration`` samples wholly inside the series."""
    duration = int(np.clip(duration, 4, length))
    start = int(rng.integers(0, length - duration + 1))
    return start, start + duration


# --------------------------------------------------------------------------------------
# Anomaly injectors
# --------------------------------------------------------------------------------------
# Signature: (rng, values, ctx) -> list[AnomalySegment]; ``values`` is modified in place.
# Labels are derived from the returned segments, never written by the injector.

@dataclass(frozen=True)
class InjectionContext:
    """Everything an injector is allowed to know about the series it perturbs."""

    scale: float
    severity: float
    target_points: int
    base_type: str
    base_params: dict

    @property
    def period(self) -> Optional[int]:
        p = self.base_params.get("period")
        return int(p) if p else None


def _inject_point(rng: np.random.Generator, values: np.ndarray, ctx: InjectionContext) -> list[AnomalySegment]:
    """Isolated global outliers with a guaranteed minimum spacing.

    Positions are drawn in a compressed index space and re-expanded, which enforces the
    spacing exactly - adjacent spikes would otherwise read as a collective anomaly.
    """
    length = values.size
    gap = POINT_MIN_SPACING
    max_points = max(1, (length - 1) // (gap + 1))
    n_points = int(min(max(1, ctx.target_points), max_points))

    free = length - (n_points - 1) * gap
    base = np.sort(rng.choice(free, size=n_points, replace=False))
    positions = base + np.arange(n_points) * gap

    signs = np.where(rng.random(n_points) < 0.5, -1.0, 1.0)
    values[positions] += signs * ctx.severity * ctx.scale
    return [
        AnomalySegment("point", int(p), int(p) + 1, ctx.severity, {"sign": int(s)})
        for p, s in zip(positions, signs)
    ]


def _alien_waveform(rng: np.random.Generator, duration: int) -> np.ndarray:
    """A unit-amplitude waveform that no base process in this module can produce."""
    shape = ("square", "triangle", "sawtooth")[int(rng.integers(3))]
    cycles = float(rng.uniform(2.0, 6.0))
    frac = np.linspace(0.0, cycles, duration, endpoint=False) % 1.0
    if shape == "square":
        return np.where(frac < 0.5, 1.0, -1.0)
    if shape == "triangle":
        return 4.0 * np.abs(frac - 0.5) - 1.0
    return 2.0 * frac - 1.0  # sawtooth


def _inject_group(rng: np.random.Generator, values: np.ndarray, ctx: InjectionContext) -> list[AnomalySegment]:
    """Shapelet anomaly: an alien deterministic waveform is laid over the segment.

    Unlike ``variance`` (resamples noise) and ``level_shift`` (moves the mean), here the
    *shape* of the subsequence stops belonging to the generating process.
    """
    start, end = _draw_segment(rng, values.size, ctx.target_points)
    duration = end - start
    w = taper(duration, max(2, duration // 10))
    values[start:end] += w * ctx.severity * ctx.scale * _alien_waveform(rng, duration)
    return [AnomalySegment("group", start, end, ctx.severity, {"duration": duration})]


def _inject_level_shift(rng: np.random.Generator, values: np.ndarray, ctx: InjectionContext) -> list[AnomalySegment]:
    """Sustained mean step with deliberately sharp (2-sample) edges."""
    start, end = _draw_segment(rng, values.size, ctx.target_points)
    duration = end - start
    shift = ctx.severity * ctx.scale * (1.0 if rng.random() < 0.5 else -1.0)
    values[start:end] += taper(duration, 2) * shift
    return [AnomalySegment("level_shift", start, end, ctx.severity, {"shift": shift, "duration": duration})]


def _inject_variance(rng: np.random.Generator, values: np.ndarray, ctx: InjectionContext) -> list[AnomalySegment]:
    """Mean-preserving inflation of local dispersion.

    The local level is estimated with a centred moving average, so the routine is safe on
    trending bases; fluctuations around it are resampled at an inflated scale.
    """
    start, end = _draw_segment(rng, values.size, ctx.target_points)
    duration = end - start
    w = taper(duration, max(2, duration // 8))

    segment = values[start:end]
    window = max(3, min(duration // 4, 51) | 1)  # odd length keeps the average centred
    kernel = np.ones(window) / window
    level = np.convolve(np.pad(segment, window // 2, mode="edge"), kernel, mode="valid")[:duration]

    inflation = 1.0 + ctx.severity * 0.5
    fluctuation = segment - level
    resampled = rng.normal(0.0, max(float(np.std(fluctuation)), ctx.scale) * inflation, duration)
    resampled -= resampled.mean()  # keep the segment mean exactly where it was

    values[start:end] = level + (1.0 - w) * fluctuation + w * resampled
    return [AnomalySegment("variance", start, end, ctx.severity, {"inflation": inflation, "duration": duration})]


def _inject_trend(rng: np.random.Generator, values: np.ndarray, ctx: InjectionContext) -> list[AnomalySegment]:
    """Transient linear drift that peaks inside the segment and returns to zero by its end.

    Returning to zero keeps the label honest: a drift left hanging at the edge would put an
    unlabelled discontinuity right after the labelled span.
    """
    start, end = _draw_segment(rng, values.size, ctx.target_points)
    duration = end - start

    knee = int(np.clip(int(duration * float(rng.uniform(0.6, 0.85))), 1, duration - 2))
    profile = np.empty(duration, dtype=np.float64)
    profile[:knee] = np.linspace(0.0, 1.0, knee, endpoint=False)
    profile[knee:] = np.linspace(1.0, 0.0, duration - knee)  # endpoint included -> ends at 0

    drift = ctx.severity * ctx.scale * (1.0 if rng.random() < 0.5 else -1.0)
    values[start:end] += profile * drift
    return [AnomalySegment("trend", start, end, ctx.severity, {"peak_drift": drift, "duration": duration})]


def _inject_seasonality(rng: np.random.Generator, values: np.ndarray, ctx: InjectionContext) -> list[AnomalySegment]:
    """Distort the seasonal component by warping its period in place.

    The original oscillation is subtracted and replaced at a perturbed period - a genuine
    frequency change, not an extra sine on top. Non-seasonal bases get an alien
    oscillation instead, since there is nothing to warp.
    """
    start, end = _draw_segment(rng, values.size, ctx.target_points)
    duration = end - start
    w = taper(duration, max(2, duration // 8))
    t = np.arange(start, end, dtype=np.float64)
    direction = 1.0 if rng.random() < 0.5 else -1.0

    period = ctx.period
    if period is None:
        alien_period = float(rng.uniform(8.0, 40.0))
        values[start:end] += w * ctx.severity * ctx.scale * np.sin(2.0 * np.pi * t / alien_period)
        return [
            AnomalySegment(
                "seasonality", start, end, ctx.severity,
                {"mode": "injected", "alien_period": alien_period, "duration": duration},
            )
        ]

    amp = float(ctx.base_params["amp"])
    phase = float(ctx.base_params["phase"])
    harmonic = float(ctx.base_params.get("harmonic", 0.0))
    # Warp the period by a severity-scaled amount, clipped so the result stays resolvable.
    warp = float(np.clip(1.0 + ctx.severity * 0.12 * direction, 0.4, 2.2))
    new_period = period * warp

    def wave(p: float) -> np.ndarray:
        return amp * np.sin(2.0 * np.pi * t / p + phase) + amp * harmonic * np.sin(
            4.0 * np.pi * t / p + phase
        )

    values[start:end] += w * (wave(new_period) - wave(float(period)))
    return [
        AnomalySegment(
            "seasonality", start, end, ctx.severity,
            {"mode": "warped", "period": period, "new_period": new_period, "duration": duration},
        )
    ]


ANOMALY_INJECTORS: dict[
    str, Callable[[np.random.Generator, np.ndarray, InjectionContext], list[AnomalySegment]]
] = {
    "point": _inject_point,
    "group": _inject_group,
    "level_shift": _inject_level_shift,
    "variance": _inject_variance,
    "trend": _inject_trend,
    "seasonality": _inject_seasonality,
}


# --------------------------------------------------------------------------------------
# Series assembly
# --------------------------------------------------------------------------------------

def generate_series(index: int, config: PoolConfig, rng: np.random.Generator) -> SeriesResult:
    """Generate one labelled series from its own random stream."""
    lo, hi = config.length_range
    length = int(rng.integers(lo, hi + 1)) if hi > lo else lo

    base_type = config.base_types[int(rng.integers(len(config.base_types)))]
    values, base_params = BASE_PROCESSES[base_type](rng, length)
    values = np.asarray(values, dtype=np.float64)

    labels = np.zeros(length, dtype=np.int8)
    segments: list[AnomalySegment] = []
    anomaly_type = "none"
    severity = 0.0
    target_fraction = 0.0

    if rng.random() < config.anomaly_rate:
        anomaly_type = config.allowed_anomaly_types[int(rng.integers(len(config.allowed_anomaly_types)))]
        f_lo, f_hi = config.anomaly_fraction_range
        target_fraction = float(rng.uniform(f_lo, f_hi))
        severity = float(rng.uniform(*config.severity_range))

        ctx = InjectionContext(
            # Measured on the clean series, so magnitudes stay comparable across pools.
            scale=local_scale(values),
            severity=severity,
            target_points=max(4, int(round(length * target_fraction))),
            base_type=base_type,
            base_params=base_params,
        )
        segments = ANOMALY_INJECTORS[anomaly_type](rng, values, ctx)
        for seg in segments:
            labels[seg.start:seg.end] = 1

    num_anomalies = int(labels.sum())
    return SeriesResult(
        series_id=f"{config.group}__SYNTH__{base_type}-{index:05d}_full",
        values=values,
        labels=labels,
        metadata={
            "series_id": f"{config.group}__SYNTH__{base_type}-{index:05d}_full",
            "length": length,
            "num_point_anomalies": num_anomalies,
            "y_i": int(num_anomalies > 0),
            "is_split": False,
            "original_length": length,
            "source_notes": f"synthetic;base={base_type};anomaly={anomaly_type};gen={__version__}",
            "base_type": base_type,
            "anomaly_type": anomaly_type,
            "anomaly_fraction": round(num_anomalies / length, 6),
            "target_fraction": round(target_fraction, 6),
            "severity": round(severity, 4),
            "num_segments": len(segments),
            "base_params": json.dumps({k: _jsonable(v) for k, v in base_params.items()}),
            "segments": json.dumps([s.as_dict() for s in segments]),
        },
    )


# --------------------------------------------------------------------------------------
# Pool assembly and validation
# --------------------------------------------------------------------------------------

def validate_pool(data: pd.DataFrame, meta: pd.DataFrame) -> None:
    """Assert every invariant the downstream pipeline relies on, before anything is written."""
    missing = {"series_id", "time_index", "value", "label"} - set(data.columns)
    if missing:
        raise AssertionError(f"data is missing columns: {sorted(missing)}")
    missing = {
        "series_id", "length", "num_point_anomalies", "y_i", "is_split", "original_length",
    } - set(meta.columns)
    if missing:
        raise AssertionError(f"metadata is missing columns: {sorted(missing)}")

    for column, expected in (("value", np.float64), ("time_index", np.int64), ("label", np.int8)):
        if data[column].dtype != expected:
            raise AssertionError(f"{column} must be {expected.__name__}, got {data[column].dtype}")

    if meta["series_id"].duplicated().any():
        dupes = meta.loc[meta["series_id"].duplicated(), "series_id"].tolist()[:5]
        raise AssertionError(f"duplicate series_id in metadata, e.g. {dupes}")
    if not np.isfinite(data["value"].to_numpy()).all():
        raise AssertionError("value contains NaN or inf")
    if not data["label"].isin((0, 1)).all():
        raise AssertionError("label must be binary")

    observed = data.groupby("series_id", sort=False).agg(
        obs_length=("time_index", "size"),
        obs_anomalies=("label", "sum"),
        first_index=("time_index", "min"),
        last_index=("time_index", "max"),
    )
    joined = meta.set_index("series_id").join(observed, how="left")

    if joined["obs_length"].isna().any():
        raise AssertionError("metadata references a series_id absent from the data frame")
    if not (joined["obs_length"] == joined["length"]).all():
        raise AssertionError("metadata length disagrees with the number of rows")
    if not (joined["obs_anomalies"] == joined["num_point_anomalies"]).all():
        raise AssertionError("metadata num_point_anomalies disagrees with the labels")
    if not (joined["y_i"] == (joined["obs_anomalies"] > 0).astype(int)).all():
        raise AssertionError("y_i disagrees with the point-wise labels")
    if not (joined["first_index"] == 0).all():
        raise AssertionError("time_index must start at 0 for every series")
    if not (joined["last_index"] == joined["length"] - 1).all():
        raise AssertionError("time_index must be contiguous within every series")


def generate_pool(config: PoolConfig, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate, validate and persist one pool."""
    config.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== %s: generating %d series ===", config.group, config.num_series)
    # crc32, not hash(): the builtin hash of a str is randomised per interpreter run and
    # would silently break reproducibility across processes.
    root = np.random.SeedSequence(config.seed, spawn_key=(zlib.crc32(config.group.encode()),))
    streams = root.spawn(config.num_series)

    results: list[SeriesResult] = []
    for i, seed_seq in enumerate(streams):
        results.append(generate_series(i, config, np.random.default_rng(seed_seq)))
        if (i + 1) % 250 == 0:
            logger.info("  ... %d/%d", i + 1, config.num_series)

    lengths = np.fromiter((r.values.size for r in results), dtype=np.int64, count=len(results))
    data = pd.DataFrame(
        {
            "series_id": np.repeat(np.array([r.series_id for r in results], dtype=object), lengths),
            "time_index": np.concatenate([np.arange(n, dtype=np.int64) for n in lengths]),
            "value": np.concatenate([r.values for r in results]).astype(np.float64),
            "label": np.concatenate([r.labels for r in results]).astype(np.int8),
        }
    )
    meta = pd.DataFrame([r.metadata for r in results])

    validate_pool(data, meta)

    data.to_parquet(output_dir / f"{config.group}.parquet", index=False)
    meta.to_parquet(output_dir / f"{config.group}_metadata.parquet", index=False)
    manifest = {
        **config.to_manifest(),
        "num_rows": int(len(data)),
        "anomalous_series": int(meta["y_i"].sum()),
        "mean_anomaly_fraction": round(float(meta["anomaly_fraction"].mean()), 6),
    }
    (output_dir / f"{config.group}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    _log_summary(config.group, meta)
    logger.info("Saved to %s/%s.parquet", output_dir, config.group)
    return data, meta


def _log_summary(group: str, meta: pd.DataFrame) -> None:
    anomalous = meta[meta["y_i"] == 1]
    logger.info("=== %s COMPLETE ===", group)
    logger.info("Series: %d | anomalous: %d (%.1f%%)", len(meta), len(anomalous), 100 * meta["y_i"].mean())
    logger.info(
        "Length min/mean/max: %d / %.1f / %d",
        meta["length"].min(), meta["length"].mean(), meta["length"].max(),
    )
    if not anomalous.empty:
        logger.info(
            "Anomalous-point fraction (dirty series) min/mean/max: %.4f / %.4f / %.4f",
            anomalous["anomaly_fraction"].min(),
            anomalous["anomaly_fraction"].mean(),
            anomalous["anomaly_fraction"].max(),
        )
        logger.info("Anomaly types: %s", anomalous["anomaly_type"].value_counts().to_dict())
    logger.info("Base processes: %s", meta["base_type"].value_counts().to_dict())


def main() -> None:
    """Build every pool listed in GROUPS_TO_BUILD."""
    for group in GROUPS_TO_BUILD:
        generate_pool(pool_config(group), OUTPUT_DIR)


if __name__ == "__main__":
    main()
