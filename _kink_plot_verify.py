"""
Generate kink diagnostic plots for every spike in tm-5-3-2, saved to a new
folder (kink_verify_NEW/) inside the bundle.  Does NOT touch any existing files.

For each spike:
  - DETECTED:  green title, kink marker shown
  - MISSED:    red title, d2-valley candidate position shown + gate failure reason
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).parent))
from kink_detection import (
    measure_kink_for_spike,
    should_skip_kink_detection_struggling_cell,
    KINK_RATIO_THRESHOLD,
    KINK_DEPTH_RATIO_THRESHOLD,
    KINK_MIN_INTERVAL_MS,
)
from analysis_config import (
    THRESHOLD_PERCENT,
    PEAK_HEIGHT_THRESHOLD,
    PEAK_PROMINENCE,
    MIN_PEAK_DISTANCE_S,
    MIN_PEAK_THRESHOLD_AMPLITUDE_MV,
    PRE_THRESHOLD_WINDOW_S,
    POST_THRESHOLD_WINDOW_S,
)

BUNDLE = (
    r"Z:\Manos\SNF_Center Data - Manos\2. Electrophysiology\SNF_Center\Human_dataR"
    r"\Developmental human study\sub-200123\sub-200123_ses-2001231tm-5-3-2_icephys"
)
TARGET_SWEEPS = [17, 18, 19, 20]  # sweeps with spiking activity
OUTPUT_SUBDIR = "kink_verify_NEW"


def _find_d2_valley(dvdt_window):
    """Return (valley_idx, depth_ratio, kink_ratio) in dvdt_window coords, or None."""
    max_upstroke_idx    = int(np.argmax(dvdt_window))
    max_upstroke_height = dvdt_window[max_upstroke_idx]
    if max_upstroke_height <= 0:
        return None
    search = dvdt_window[0:max_upstroke_idx]
    if len(search) < 3:
        return None
    d2 = np.diff(search)
    if len(d2) < 3:
        return None
    valley_idx = None
    for i in range(1, len(d2) - 1):
        if d2[i] < d2[i - 1] and d2[i] < d2[i + 1]:
            if valley_idx is None or d2[i] < d2[valley_idx]:
                valley_idx = i
    if valley_idx is None:
        return None
    d2_left  = np.max(d2[:valley_idx])     if valley_idx > 0          else d2[0]
    d2_right = np.max(d2[valley_idx + 1:]) if valley_idx < len(d2) - 1 else d2[-1]
    d2_norm  = max(d2_left, d2_right)
    if d2_norm <= 0:
        return None
    depth_ratio  = (d2_norm - d2[valley_idx]) / d2_norm
    kink_ratio   = dvdt_window[valley_idx] / max_upstroke_height
    interval_ms  = None  # computed by caller with time array
    return {"valley_idx": valley_idx, "depth_ratio": depth_ratio, "kink_ratio": kink_ratio,
            "max_upstroke_idx": max_upstroke_idx}


def plot_spike(kink_v, kink_t, kink_dvdt, detected, spike_label, valley_info, metrics, out_dir):
    """
    Two-panel plot (voltage + dV/dt) for one spike's kink window.
    detected=True  → green title, kink marker at valley_idx
    detected=False → red title, candidate marker at valley_idx + gate failure text
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    t_rel = (kink_t - kink_t[0]) * 1000  # ms from threshold

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    color = "darkgreen" if detected else "firebrick"
    status = "KINK DETECTED" if detected else "NO KINK DETECTED"

    # Gate failure text for missed spikes
    if not detected and valley_info is not None:
        fails = []
        if valley_info["kink_ratio"]  < KINK_RATIO_THRESHOLD:
            fails.append(f"ratio={valley_info['kink_ratio']:.3f} < {KINK_RATIO_THRESHOLD}")
        if valley_info["depth_ratio"] < KINK_DEPTH_RATIO_THRESHOLD:
            fails.append(f"depth={valley_info['depth_ratio']:.3f} < {KINK_DEPTH_RATIO_THRESHOLD}")
        fail_str = " | ".join(fails) if fails else "no valley found"
        title = f"{spike_label} — {status}\n{fail_str}"
    elif detected:
        title = (f"{spike_label} — {status}\n"
                 f"ratio={metrics['kink_ratio']:.3f}  interval={metrics['kink_interval_ms']:.2f}ms")
    else:
        title = f"{spike_label} — {status}\n(no d2 valley found in search window)"

    n = len(kink_v)

    # --- Voltage panel ---
    axes[0].plot(t_rel, kink_v, "k-", lw=1.5)
    axes[0].scatter(t_rel[0], kink_v[0], s=80, color="green", zorder=5, label="Threshold", marker="o")
    # mark max upstroke position in voltage
    up_local = int(np.argmax(kink_dvdt))
    axes[0].scatter(t_rel[up_local], kink_v[up_local], s=80, color="red", zorder=5, label="Max upstroke", marker="^")
    # mark valley/kink position
    if valley_info is not None:
        vi = valley_info["valley_idx"]
        if 0 <= vi < n:
            marker_color = "orange" if detected else "royalblue"
            marker_label = "Kink" if detected else "Candidate (failed)"
            axes[0].scatter(t_rel[vi], kink_v[vi], s=100, color=marker_color, zorder=6,
                            label=marker_label, marker="s")
    axes[0].axvspan(t_rel[0], t_rel[-1], alpha=0.07, color="yellow")
    axes[0].set_ylabel("Voltage (mV)", fontsize=10, fontweight="bold")
    axes[0].set_title(title, fontsize=10, fontweight="bold", color=color)
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # --- dV/dt panel ---
    axes[1].plot(t_rel, kink_dvdt, color="purple", lw=1.5, label="dV/dt")
    axes[1].scatter(t_rel[0], kink_dvdt[0], s=80, color="green", zorder=5, marker="o")
    axes[1].scatter(t_rel[up_local], kink_dvdt[up_local], s=80, color="red", zorder=5, marker="^")
    if valley_info is not None:
        vi = valley_info["valley_idx"]
        if 0 <= vi < n:
            marker_color = "orange" if detected else "royalblue"
            axes[1].scatter(t_rel[vi], kink_dvdt[vi], s=100, color=marker_color, zorder=6, marker="s")
            # draw a vertical dashed line at the candidate
            axes[1].axvline(t_rel[vi], color=marker_color, lw=1, ls="--", alpha=0.6)
    axes[1].axvspan(t_rel[0], t_rel[-1], alpha=0.07, color="yellow")
    axes[1].set_ylabel("dV/dt (mV/ms)", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("Time from threshold (ms)", fontsize=10, fontweight="bold")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = out_dir / f"{spike_label.replace(' ', '_')}.jpeg"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()


def run():
    p = Path(BUNDLE)
    out_dir = p / OUTPUT_SUBDIR
    print(f"Output folder: {out_dir}")

    manifest = json.loads((p / "manifest.json").read_text())
    fs_raw = manifest.get("meta", {}).get("sampleRate_Hz", 50000)
    fs = float(fs_raw) if not isinstance(fs_raw, list) else max(float(x) for x in fs_raw)

    mv_files = list(p.rglob("mV_*_clean.parquet")) or list(p.rglob("mV_*.parquet"))
    if not mv_files:
        print("ERROR: no mV parquet found"); return
    df = pd.read_parquet(mv_files[0]).sort_values(["sweep", "t_s"])

    sc_path = p / "sweep_config.json"
    sweep_config = json.loads(sc_path.read_text()) if sc_path.exists() else {}

    t_stim_start = t_stim_end = None
    for sid, sdata in sweep_config.get("sweeps", {}).items():
        if sdata.get("valid", False):
            w = sdata.get("windows", {})
            t_stim_start = w.get("stimulus_start_s")
            t_stim_end   = w.get("stimulus_end_s")
            break
    if t_stim_start is None:
        print("ERROR: no valid sweep in sweep_config"); return

    pre_s, post_s = PRE_THRESHOLD_WINDOW_S, POST_THRESHOLD_WINDOW_S
    total = detected_total = 0

    for sweep_number in sorted(df["sweep"].unique()):
        if sweep_number not in TARGET_SWEEPS:
            continue

        group = df[df["sweep"] == sweep_number].sort_values("t_s")
        time    = group["t_s"].to_numpy()
        voltage = group["value"].to_numpy()

        sc_sw    = sweep_config.get("sweeps", {}).get(str(sweep_number), {})
        sw_start = sc_sw["windows"]["stimulus_start_s"] if sc_sw and "windows" in sc_sw else t_stim_start
        sw_end   = sc_sw["windows"]["stimulus_end_s"]   if sc_sw and "windows" in sc_sw else t_stim_end

        mask = (time >= sw_start - pre_s) & (time <= sw_end + post_s)
        time = time[mask];  voltage = voltage[mask]
        if len(time) < 10:
            continue

        peaks, _ = find_peaks(voltage, height=PEAK_HEIGHT_THRESHOLD, prominence=PEAK_PROMINENCE)
        filtered = []
        for idx in peaks:
            if not filtered or (time[idx] - time[filtered[-1]]) >= MIN_PEAK_DISTANCE_S:
                filtered.append(idx)
        peaks = np.array(filtered, dtype=int)
        peaks = peaks[(time[peaks] >= sw_start) & (time[peaks] <= sw_end)]
        if len(peaks) == 0:
            continue

        base_mask = (time >= sw_start - 0.05) & (time < sw_start)
        v_resting = float(np.median(voltage[base_mask])) if base_mask.any() else float(np.median(voltage[:100]))
        spike_amplitudes = [float(voltage[int(pk)]) - v_resting for pk in peaks]

        print(f"\nSweep {sweep_number}: {len(peaks)} spikes")

        for i, peak in enumerate(peaks):
            peak = int(peak)
            t_peak = float(time[peak]);  v_peak = float(voltage[peak])
            t_w1_start = max(time[peak] - pre_s, float(time[0]))
            w1_start = int(np.searchsorted(time, t_w1_start, side="left"))
            w1_end   = peak + 1
            if w1_start >= w1_end:
                continue
            t_up = time[w1_start:w1_end];  v_up = voltage[w1_start:w1_end]
            dvdt_up = np.gradient(v_up, t_up) * 1000
            if len(dvdt_up) == 0:
                continue
            max_dvdt = float(np.max(dvdt_up))
            up_rel   = int(np.argmax(dvdt_up))
            thr_val  = THRESHOLD_PERCENT * max_dvdt
            below    = np.where(dvdt_up >= thr_val)[0]
            if len(below) == 0:
                continue
            thr_rel = int(below[0])
            if v_peak - float(v_up[thr_rel]) < MIN_PEAK_THRESHOLD_AMPLITUDE_MV:
                continue

            kink_v    = v_up[thr_rel:up_rel + 1]
            kink_t    = t_up[thr_rel:up_rel + 1]
            kink_dvdt = dvdt_up[thr_rel:up_rel + 1]

            if should_skip_kink_detection_struggling_cell(spike_amplitudes, i):
                print(f"  spike#{i+1}: skipped (struggling cell)")
                continue

            metrics     = measure_kink_for_spike(kink_v, kink_t, kink_dvdt)
            detected    = metrics["num_kinks"] > 0
            valley_info = _find_d2_valley(kink_dvdt)

            label = f"sw{sweep_number}_spike{i+1:02d}"
            plot_spike(kink_v, kink_t, kink_dvdt, detected, label, valley_info, metrics, out_dir)

            status = "KINK" if detected else "miss"
            if valley_info:
                print(f"  spike#{i+1}: {status}  ratio={valley_info['kink_ratio']:.3f}  "
                      f"depth={valley_info['depth_ratio']:.3f}  n={len(kink_dvdt)}samp")
            else:
                print(f"  spike#{i+1}: {status}  (no valley)")
            total += 1
            if detected:
                detected_total += 1

    print(f"\nTotal: {detected_total}/{total} detected ({100*detected_total/total:.0f}%)")
    print(f"Plots saved to: {out_dir}")


if __name__ == "__main__":
    run()
