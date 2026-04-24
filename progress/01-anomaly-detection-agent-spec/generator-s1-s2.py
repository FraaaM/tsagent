import numpy as np
import pandas as pd
from pathlib import Path
import random
from typing import Tuple, Dict, Optional, List, Union

np.random.seed(42)
random.seed(42)

# Базовые процессы
def generate_base_series(length: int, base_type: str, params: Optional[Dict] = None) -> np.ndarray:
    if params is None:
        params = {}
    
    if base_type == "white_noise":
        return np.random.normal(0, params.get("sigma", 0.7), length)
    
    elif base_type == "ar1":
        phi = params.get("phi", 0.8)
        sigma = params.get("sigma", 0.7)
        series = np.zeros(length)
        series[0] = np.random.normal(0, sigma)
        for t in range(1, length):
            series[t] = phi * series[t-1] + np.random.normal(0, sigma)
        return series
    
    elif base_type == "ar2":                                      
        phi1 = params.get("phi1", 0.8)
        phi2 = params.get("phi2", -0.3)
        sigma = params.get("sigma", 0.7)
        series = np.zeros(length)
        series[0] = np.random.normal(0, sigma)
        series[1] = np.random.normal(0, sigma)
        for t in range(2, length):
            series[t] = (phi1 * series[t-1] + phi2 * series[t-2] +
                        np.random.normal(0, sigma))
        return series
    
    elif base_type == "pink_noise":                              
        freqs = np.fft.fftfreq(length)
        freqs[0] = 1.0
        magnitudes = 1.0 / np.sqrt(np.abs(freqs))
        phases = np.random.uniform(0, 2*np.pi, length)
        noise = np.fft.ifft(magnitudes * np.exp(1j * phases)).real
        return noise / np.std(noise) * params.get("sigma", 1.0)
    
    elif base_type == "linear_trend":
        slope = params.get("slope", 0.01)
        sigma = params.get("sigma", 0.5)
        t = np.arange(length)
        return slope * t + np.random.normal(0, sigma, length)
    
    elif base_type == "seasonal_sine":
        freq = params.get("freq", 0.05)
        amp = params.get("amp", 1.5)
        sigma = params.get("sigma", 0.5)
        t = np.arange(length)
        return amp * np.sin(2 * np.pi * freq * t) + np.random.normal(0, sigma, length)
    
    elif base_type == "random_walk":
        sigma = params.get("sigma", 1.0)
        return np.cumsum(np.random.normal(0, sigma, length))
    
    elif base_type == "trend_seasonal":                           
        slope = params.get("slope", 0.008)
        freq = params.get("freq", 0.04)
        amp = params.get("amp", 2.5)
        sigma = params.get("sigma", 0.6)
        t = np.arange(length)
        return (slope * t +
                amp * np.sin(2 * np.pi * freq * t) +
                np.random.normal(0, sigma, length))
    
    else:
        raise ValueError(f"Неизвестный base_type: {base_type}")


# Инжекторы аномалий
def inject_anomaly(series: np.ndarray, 
                   anomaly_type: str, 
                   severity: float,
                   target_points: int,
                   use_local_std: bool = False,      
                   local_window: int = 50,         
                   ramp_length: int = 8) -> Tuple[np.ndarray, np.ndarray]:

    length = len(series)
    series = series.copy()
    label = np.zeros(length, dtype=np.int8)

    # точечные аномалии
    if anomaly_type == "point":
        num_points = max(1, min(target_points//3, 20))
        positions = np.random.choice(length, num_points, replace=False)
        
        for pos in positions:
            if use_local_std and length > local_window * 2:
                start = max(0, pos - local_window)
                end = min(length, pos + local_window)
                local_std = np.std(series[start:end])
                if local_std < 1e-8:
                    local_std = np.std(series)
                deviation = severity * local_std
            else:
                deviation = severity * np.std(series) * 1.1
            
            series[pos] += random.choice([-1, 1]) * deviation
            label[pos] = 1
            
        return series, label

    # сегментные аномалии
    duration = max(10, min(target_points, length // 4))
    start = random.randint(0, length - duration - ramp_length)
    end = start + duration

    ramp = np.linspace(0, 1, ramp_length)
    ramp_rev = ramp[::-1]

    if anomaly_type == "group":
        factor = 1 + severity * random.choice([-1, 1])
        series[start:end] *= factor

    elif anomaly_type == "level_shift":
        std = np.std(series)
        shift = severity * std * random.choice([-1, 1])
        series[start:start+ramp_length] += ramp * shift
        series[start+ramp_length:end-ramp_length] += shift
        series[end-ramp_length:end] += ramp_rev * shift

    elif anomaly_type == "trend":
        extra_slope = severity * 0.08 * random.choice([-1, 1])
        t_local = np.arange(duration)
        series[start:end] += extra_slope * t_local

    elif anomaly_type == "variance":
        orig_mean = np.mean(series[start:end])
        new_std = np.std(series[start:end]) * (1 + severity * random.choice([0.5, 0.8]))
        noise = np.random.normal(0, new_std, duration)
        series[start:end] = orig_mean + noise

    elif anomaly_type == "seasonality":
        t_local = np.arange(duration)
        amp = severity * random.uniform(1.0, 1.2)
        freq = random.uniform(0.06, 0.18)
        extra = amp * np.sin(2 * np.pi * freq * t_local)
        series[start:end] += extra

    label[start:end] = 1
    return series, label


def generate_single_series(series_idx: int, group: str, base_type: str,
                           length: int,
                           anomaly_rate: float = 0.5,
                           allowed_anomaly_types: Optional[List[str]] = None,
                           target_anomaly_fraction: Optional[Union[float, Tuple[float, float]]] = None,
                           severity_range: Tuple[float, float] = (2.0, 5.0)) -> Tuple[pd.DataFrame, Dict]:
    
    if isinstance(length, tuple):
        length = random.randint(length[0], length[1])
    
    params = {}
    # if base_type in ["ar1", "ar2"]:
    #     params["phi"] = round(random.uniform(0.4, 0.9), 2) if base_type == "ar1" else None
    # elif base_type == "linear_trend":
    #     params["slope"] = round(random.uniform(0.005, 0.03), 4)
    # elif base_type in ["seasonal_sine", "trend_seasonal"]:
    #     params["freq"] = round(random.uniform(0.02, 0.12), 4)
    #     params["amp"] = round(random.uniform(1.5, 3.5), 2)
    
    clean_series = generate_base_series(length, base_type) #params
    is_anomalous = random.random() < anomaly_rate
    
    if not is_anomalous:
        anomaly_type = "none"
        final_series = clean_series.copy()
        label = np.zeros(length, dtype=np.int8)
    else:
        all_types = ["point", "group", "level_shift", "trend", "variance", "seasonality"]
        if allowed_anomaly_types is None:
            allowed_anomaly_types = all_types
        anomaly_type = random.choice([t for t in allowed_anomaly_types if t in all_types])
        
        if target_anomaly_fraction is None:
            target_fraction = random.uniform(0.01, 0.12)
        elif isinstance(target_anomaly_fraction, tuple):
            target_fraction = random.uniform(target_anomaly_fraction[0], target_anomaly_fraction[1])
        else:
            target_fraction = target_anomaly_fraction
        
        target_points = max(3, int(length * target_fraction * random.uniform(0.8, 1.2)))
        severity = random.uniform(severity_range[0], severity_range[1])
        final_series, label = inject_anomaly(clean_series, anomaly_type, severity, target_points)
    
    df = pd.DataFrame({
        "series_id": [f"{group}__SYNTH__{base_type}__{series_idx}_full"] * length,
        "time_index": np.arange(length),
        "value": final_series.astype(np.float64),
        "label": label
    })
    
    actual_fraction = label.mean()
    
    metadata = {
        "series_id": df["series_id"].iloc[0],
        "length": length,
        "num_point_anomalies": int(label.sum()),
        "anomaly_fraction": round(actual_fraction, 4),
        "y_i": 1 if label.sum() > 0 else 0,
        "is_split": False,
        "original_length": length,
        "base_type": base_type,
        "anomaly_type": anomaly_type,
        "severity": round(severity, 2) if is_anomalous else 0.0,
        "target_fraction": target_fraction if is_anomalous else 0.0
    }
    
    return df, metadata


def generate_synthetic_pool(group: str, num_series: int, length: int,
                            base_types: List[str],
                            anomaly_rate: float = 0.5,
                            allowed_anomaly_types: Optional[List[str]] = None,
                            target_anomaly_fraction: Optional[Union[float, Tuple[float, float]]] = None,
                            severity_range: Tuple[float, float] = (2.0, 5.0),
                            output_dir: str = "data/synthetic") -> None:
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_data = []
    all_metadata = []
    
    print(f"Генерация {num_series} серий для {group}...")
    for i in range(num_series):
        base_type = random.choice(base_types)
        df_series, meta = generate_single_series(
            i, group, base_type, length,
            anomaly_rate, allowed_anomaly_types,
            target_anomaly_fraction, severity_range
        )
        all_data.append(df_series)
        all_metadata.append(meta)
        
        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{num_series}")
    
    full_data = pd.concat(all_data, ignore_index=True)
    metadata_df = pd.DataFrame(all_metadata)
    
    full_data.to_parquet(output_dir / f"{group}.parquet", index=False)
    metadata_df.to_parquet(output_dir / f"{group}_metadata.parquet", index=False)
    
    print(f"{group} готово")
    print(f"Аномальных серий: {metadata_df['y_i'].mean():.1%}")
    print(f"Средняя доля аномальных точек: {metadata_df['anomaly_fraction'].mean():.2%}")


if __name__ == "__main__":

    #  S1 — Stationary (только стационарные процессы) 
    generate_synthetic_pool(
        group="S1",
        num_series=500,                             # количество рядов
        length=1000,                                # длина каждого ряда
        base_types=["white_noise", "ar1", "ar2"],   # алгоритмы создания ряда
        anomaly_rate=0.5,                           # доля аномалиьных рядов в пуле
        allowed_anomaly_types=["point", "group", "level_shift", "variance"], # типы аномалий
        target_anomaly_fraction=(0.025, 0.05),      # доля аномальных точек в ряде
        severity_range=(2.6, 4),                    # кол-во std отклонений выброса от среднего
        output_dir="data/synthetic"                 # путь для генерации данных
    )
    
    # S2 — Trend-Seasonal 
    generate_synthetic_pool(
        group="S2",
        num_series=500,
        length=1000,
        base_types=["linear_trend", "seasonal_sine", "trend_seasonal"],
        anomaly_rate=0.5,
        allowed_anomaly_types=["trend", "seasonality", "group", "level_shift"],
        target_anomaly_fraction=(0.025, 0.05),   
        severity_range=(2.6, 4),
        output_dir="data/synthetic"
    )