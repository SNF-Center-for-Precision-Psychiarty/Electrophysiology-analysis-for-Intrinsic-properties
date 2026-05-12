"""Debug kink detection for sub-131113 sweep17 peak17"""
import sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).parent))
from kink_detection import measure_kink_for_spike
from analysis_config import (
    THRESHOLD_PERCENT, PEAK_HEIGHT_THRESHOLD, PEAK_PROMINENCE,
    MIN_PEAK_DISTANCE_S, MIN_PEAK_THRESHOLD_AMPLITUDE_MV,
    PRE_THRESHOLD_WINDOW_S, POST_THRESHOLD_WINDOW_S,
)

p = Path(r"Z:\Manos\SNF_Center Data - Manos\2. Electrophysiology\SNF_Center\Human_dataR\Developmental human study\sub-131113\sub-131113_ses-1311131mg-4-1-1_icephys")

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

TARGET_SWEEP = 17
TARGET_PEAK  = 17   # 0-indexed

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

print(f"Total peaks in sweep {TARGET_SWEEP}: {len(peaks)}")
if TARGET_PEAK >= len(peaks):
    print("TARGET_PEAK out of range"); sys.exit(1)

i    = TARGET_PEAK
peak = int(peaks[i])

t_w1_start = max(time[peak] - PRE_THRESHOLD_WINDOW_S, float(time[0]))
w1_start   = int(np.searchsorted(time, t_w1_start, side="left"))
w1_end     = peak + 1

t_up    = time[w1_start:w1_end]
v_up    = voltage[w1_start:w1_end]
dvdt_up = np.gradient(v_up, t_up) * 1000

max_dvdt = float(np.max(dvdt_up))
up_rel   = int(np.argmax(dvdt_up))

thr_val  = THRESHOLD_PERCENT * max_dvdt
below    = np.where(dvdt_up >= thr_val)[0]
thr_rel  = int(below[0])

kink_dvdt = dvdt_up[thr_rel:up_rel + 1]
kink_t    = t_up[thr_rel:up_rel + 1]
kink_v    = v_up[thr_rel:up_rel + 1]

print(f"\n--- Spike window ---")
print(f"thr_rel={thr_rel}, up_rel={up_rel}, window size={len(kink_dvdt)} samples")
print(f"dV/dt at threshold: {kink_dvdt[0]:.4e} mV/ms")
print(f"dV/dt max (upstroke): {max_dvdt:.4e} mV/ms")
print(f"dV/dt ratio at threshold: {kink_dvdt[0]/max_dvdt:.3f}")
print(f"\ndV/dt values (threshold→upstroke):")
for j, v in enumerate(kink_dvdt):
    t_rel_ms = (kink_t[j] - kink_t[0]) * 1000
    print(f"  [{j:3d}] t={t_rel_ms:.4f}ms  dV/dt={v:.4e}")

print(f"\n--- d2 (diff of dV/dt) ---")
d2 = np.diff(kink_dvdt)
for j, v in enumerate(d2):
    print(f"  [{j:3d}] d2={v:.4e}")

# Find deepest interior valley
valley_idx = None
for ii in range(1, len(d2) - 1):
    if d2[ii] < d2[ii - 1] and d2[ii] < d2[ii + 1]:
        if valley_idx is None or d2[ii] < d2[valley_idx]:
            valley_idx = ii

print(f"\nDeepest interior valley in d2: index={valley_idx}")
if valley_idx is not None:
    print(f"  d2[valley_idx]={d2[valley_idx]:.4e}")
    print(f"  Corresponding kink_dvdt[valley_idx]={kink_dvdt[valley_idx]:.4e}")

# Backwards snap
if valley_idx is not None:
    n_search = len(kink_dvdt)
    kink_pos_local = valley_idx
    print(f"\nBackwards snap search (from j={min(valley_idx, n_search-1)} down to 1):")
    for j in range(min(valley_idx, n_search - 1), 0, -1):
        cond = kink_dvdt[j] > kink_dvdt[j - 1] and kink_dvdt[j] >= kink_dvdt[j + 1] if j + 1 < n_search else False
        print(f"  j={j}: dvdt[j]={kink_dvdt[j]:.4e} > dvdt[j-1]={kink_dvdt[j-1]:.4e}? {kink_dvdt[j] > kink_dvdt[j-1]}  >= dvdt[j+1]={kink_dvdt[j+1] if j+1<n_search else 'N/A'}? {cond}")
        if kink_dvdt[j] > kink_dvdt[j - 1] and j + 1 < n_search and kink_dvdt[j] >= kink_dvdt[j + 1]:
            kink_pos_local = j
            print(f"  --> SNAPPED to j={j}")
            break
    print(f"Final kink_pos_local={kink_pos_local}")
    print(f"kink dV/dt={kink_dvdt[kink_pos_local]:.4e}, t_rel={(kink_t[kink_pos_local]-kink_t[0])*1000:.4f}ms")

print(f"\n--- measure_kink_for_spike (debug=True) ---")
result = measure_kink_for_spike(kink_v, kink_t, kink_dvdt, debug=True)
print(f"\nResult: {result}")
