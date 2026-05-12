"""Diagnostic plots for sub-201202 ses-2012022tm-1-1-1 sweep 8 misses"""
import sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).parent))
from kink_detection import measure_kink_metrics, KINK_RATIO_THRESHOLD, KINK_DEPTH_RATIO_THRESHOLD, KINK_MIN_INTERVAL_MS
from analysis_config import (
    THRESHOLD_PERCENT, PEAK_HEIGHT_THRESHOLD, PEAK_PROMINENCE,
    MIN_PEAK_DISTANCE_S, MIN_PEAK_THRESHOLD_AMPLITUDE_MV,
    PRE_THRESHOLD_WINDOW_S, POST_THRESHOLD_WINDOW_S,
)

p = Path(r"Z:\Manos\SNF_Center Data - Manos\2. Electrophysiology\SNF_Center\Human_dataR\Developmental human study\sub-201202\sub-201202_ses-2012022tm-1-1-1_icephys")
out_dir = p / "sweep8_debug_plots"
out_dir.mkdir(exist_ok=True)

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
    up_idx   = w1_start + up_rel

    thr_val = THRESHOLD_PERCENT * max_dvdt
    below   = np.where(dvdt_up >= thr_val)[0]
    if len(below) == 0:
        continue
    thr_rel      = int(below[0])
    threshold_idx = w1_start + thr_rel

    kink_dvdt = dvdt_up[thr_rel:up_rel + 1]
    kink_t    = t_up[thr_rel:up_rel + 1]
    kink_v    = v_up[thr_rel:up_rel + 1]

    result = measure_kink_metrics(kink_dvdt, kink_t, threshold_idx=0)
    kink_detected = result['num_kinks'] > 0

    # Determine failure reason
    if not kink_detected:
        d2 = np.diff(kink_dvdt)
        valley_idx = None
        for ii in range(1, len(d2) - 1):
            if d2[ii] < d2[ii-1] and d2[ii] < d2[ii+1]:
                if valley_idx is None or d2[ii] < d2[valley_idx]:
                    valley_idx = ii
        if valley_idx is None:
            fail_reason = "No interior valley in d2"
            candidate_local = None
        else:
            n_search = len(kink_dvdt)
            kink_pos_local = valley_idx
            for j in range(min(valley_idx, n_search - 1), 0, -1):
                if kink_dvdt[j] > kink_dvdt[j-1] and j+1 < n_search and kink_dvdt[j] >= kink_dvdt[j+1]:
                    kink_pos_local = j
                    break
            candidate_local = kink_pos_local
            cand_dvdt  = kink_dvdt[candidate_local]
            cand_ratio = cand_dvdt / max_dvdt

            d2_left  = np.max(d2[:valley_idx]) if valley_idx > 0 else d2[0]
            d2_right = np.max(d2[valley_idx+1:]) if valley_idx < len(d2)-1 else d2[-1]
            d2_norm  = max(d2_left, d2_right)
            depth    = (d2_norm - d2[valley_idx]) / d2_norm if d2_norm > 0 else 0
            interval = abs((kink_t[candidate_local] - kink_t[-1]) * 1000)

            parts = []
            if cand_ratio < KINK_RATIO_THRESHOLD:
                parts.append(f"ratio={cand_ratio:.3f} < {KINK_RATIO_THRESHOLD}")
            if depth < KINK_DEPTH_RATIO_THRESHOLD:
                parts.append(f"depth={depth:.3f} < {KINK_DEPTH_RATIO_THRESHOLD}")
            if interval < KINK_MIN_INTERVAL_MS:
                parts.append(f"interval={interval:.3f}ms < {KINK_MIN_INTERVAL_MS}ms")
            fail_reason = "FAIL: " + ", ".join(parts) if parts else "unknown"
    else:
        candidate_local = result['kink_idx']
        fail_reason = None

    # --- Plot ---
    t_rel = (kink_t - kink_t[0]) * 1000

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    title_color = 'green' if kink_detected else 'red'
    title_txt   = f"sweep{TARGET_SWEEP}_spike{i+1}  |  {'KINK' if kink_detected else 'MISS: ' + fail_reason}"

    # Voltage
    axes[0].plot(t_rel, kink_v, 'k-', lw=1.5)
    axes[0].scatter(t_rel[0], kink_v[0], s=100, color='green', zorder=5, label='Threshold', marker='o')
    axes[0].scatter(t_rel[-1], kink_v[-1], s=100, color='red', zorder=5, label='Max upstroke', marker='^')
    if candidate_local is not None and 0 <= candidate_local < len(t_rel):
        c = 'orange' if kink_detected else 'dodgerblue'
        axes[0].scatter(t_rel[candidate_local], kink_v[candidate_local],
                        s=100, color=c, zorder=5, label='Candidate kink', marker='s')
    axes[0].set_ylabel('Voltage (mV)', fontsize=11, fontweight='bold')
    axes[0].set_title(title_txt, fontsize=11, fontweight='bold', color=title_color)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # dV/dt
    axes[1].plot(t_rel, kink_dvdt, color='purple', lw=1.5, label='dV/dt')
    axes[1].axhline(KINK_RATIO_THRESHOLD * max_dvdt, color='gray', ls='--', lw=1,
                    label=f'Ratio threshold ({KINK_RATIO_THRESHOLD:.0%} of max)')
    axes[1].scatter(t_rel[-1], kink_dvdt[-1], s=100, color='red', zorder=5, marker='^')
    if candidate_local is not None and 0 <= candidate_local < len(t_rel):
        c = 'orange' if kink_detected else 'dodgerblue'
        axes[1].scatter(t_rel[candidate_local], kink_dvdt[candidate_local],
                        s=100, color=c, zorder=5, marker='s')
    axes[1].set_ylabel('dV/dt (mV/ms)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Time relative to threshold (ms)', fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / f"sweep{TARGET_SWEEP}_spike{i+1}.jpeg", dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  spike {i+1}: {title_txt}")

print(f"\nPlots saved to: {out_dir}")
