"""Debug kink detection for sub-201202 ses-2012022tm-1-1-1 sweep 8"""
import sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).parent))
from kink_detection import measure_kink_metrics
from analysis_config import (
    THRESHOLD_PERCENT, PEAK_HEIGHT_THRESHOLD, PEAK_PROMINENCE,
    MIN_PEAK_DISTANCE_S, MIN_PEAK_THRESHOLD_AMPLITUDE_MV,
    PRE_THRESHOLD_WINDOW_S, POST_THRESHOLD_WINDOW_S,
)

p = Path(r"Z:\Manos\SNF_Center Data - Manos\2. Electrophysiology\SNF_Center\Human_dataR\Developmental human study\sub-201202\sub-201202_ses-2012022tm-1-1-1_icephys")

manifest     = json.loads((p / "manifest.json").read_text())
fs_raw       = manifest.get("meta", {}).get("sampleRate_Hz", 50000)
fs           = float(fs_raw) if not isinstance(fs_raw, list) else max(float(x) for x in fs_raw)
mv_files     = list(p.rglob("mV_*_clean.parquet")) or list(p.rglob("mV_*.parquet"))
df           = pd.read_parquet(mv_files[0]).sort_values(["sweep", "t_s"])
sweep_config = json.loads((p / "sweep_config.json").read_text())

t_stim_start = t_stim_end = None
for sid, sdata in sweep_config.get("sweeps", {}).items():
    if sdata.get("valid", False):
        w = sdata.get("windows", {})
        t_stim_start = w.get("stimulus_start_s")
        t_stim_end   = w.get("stimulus_end_s")
        break

TARGET_SWEEP = 8

group   = df[df["sweep"] == TARGET_SWEEP].sort_values("t_s")
time    = group["t_s"].to_numpy()
voltage = group["value"].to_numpy()

sc_sw    = sweep_config.get("sweeps", {}).get(str(TARGET_SWEEP), {})
sw_start = sc_sw["windows"]["stimulus_start_s"] if sc_sw and "windows" in sc_sw else t_stim_start
sw_end   = sc_sw["windows"]["stimulus_end_s"]   if sc_sw and "windows" in sc_sw else t_stim_end

mask    = (time >= sw_start - PRE_THRESHOLD_WINDOW_S) & (time <= sw_end + POST_THRESHOLD_WINDOW_S)
time    = time[mask];  voltage = voltage[mask]

peaks, _ = find_peaks(voltage, height=PEAK_HEIGHT_THRESHOLD, prominence=PEAK_PROMINENCE)
filtered = []
for idx in peaks:
    if not filtered or (time[idx] - time[filtered[-1]]) >= MIN_PEAK_DISTANCE_S:
        filtered.append(idx)
peaks = np.array(filtered, dtype=int)
peaks = peaks[(time[peaks] >= sw_start) & (time[peaks] <= sw_end)]
print(f"Sweep {TARGET_SWEEP}: {len(peaks)} spikes detected\n")

from kink_detection import (KINK_RATIO_THRESHOLD, KINK_DEPTH_RATIO_THRESHOLD, KINK_MIN_INTERVAL_MS)

for i, peak in enumerate(peaks):
    peak = int(peak)
    t_w1_start = max(time[peak] - PRE_THRESHOLD_WINDOW_S, float(time[0]))
    w1_start   = int(np.searchsorted(time, t_w1_start, side="left"))
    w1_end     = peak + 1

    t_up    = time[w1_start:w1_end]
    v_up    = voltage[w1_start:w1_end]
    dvdt_up = np.gradient(v_up, t_up) * 1000

    max_dvdt = float(np.max(dvdt_up))
    up_rel   = int(np.argmax(dvdt_up))

    thr_val = THRESHOLD_PERCENT * max_dvdt
    below   = np.where(dvdt_up >= thr_val)[0]
    if len(below) == 0:
        print(f"  spike {i+1}: no threshold found"); continue
    thr_rel = int(below[0])

    kink_dvdt = dvdt_up[thr_rel:up_rel + 1]
    kink_t    = t_up[thr_rel:up_rel + 1]
    kink_v    = v_up[thr_rel:up_rel + 1]

    n_window = len(kink_dvdt)
    print(f"spike {i+1}: window={n_window} samples, max_dvdt={max_dvdt:.2e}, dvdt_at_thr={kink_dvdt[0]:.2e} ({kink_dvdt[0]/max_dvdt:.2f}x)")

    result = measure_kink_metrics(kink_dvdt, kink_t, threshold_idx=0, debug=True)
    print(f"  --> num_kinks={result['num_kinks']}, ratio={result['kink_ratio']}, depth={result.get('kink_ratio','')}, interval={result['kink_interval_ms']}")
    print()
