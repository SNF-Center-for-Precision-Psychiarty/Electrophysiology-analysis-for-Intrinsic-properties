"""Verify kink_slope_dvdt for three specific spike examples."""
import sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).parent))
from kink_detection import measure_kink_for_spike, KINK_RATIO_THRESHOLD, KINK_MIN_INTERVAL_MS, KINK_DEPTH_RATIO_THRESHOLD
from analysis_config import (
    THRESHOLD_PERCENT, PEAK_HEIGHT_THRESHOLD, PEAK_PROMINENCE,
    MIN_PEAK_DISTANCE_S, MIN_PEAK_THRESHOLD_AMPLITUDE_MV,
    PRE_THRESHOLD_WINDOW_S, POST_THRESHOLD_WINDOW_S,
)

CASES = [
    (r"Z:\Manos\SNF_Center Data - Manos\2. Electrophysiology\SNF_Center\Human_dataR\Developmental human study\sub-220218\sub-220218_ses-2202181pz-9-1-3_icephys", 11, 0),
    (r"Z:\Manos\SNF_Center Data - Manos\2. Electrophysiology\SNF_Center\Human_dataR\Developmental human study\sub-230817\sub-230817_ses-2308171si-4-10-1_icephys", 26, 0),
    (r"Z:\Manos\SNF_Center Data - Manos\2. Electrophysiology\SNF_Center\Human_dataR\Developmental human study\sub-140207\sub-140207_ses-1402072pi-1-1-2_icephys", 15, 1),
]

out_dir = Path(r"k:\abf_nwb_pipeline\slope_verify_plots")
out_dir.mkdir(exist_ok=True)

for bundle_str, target_sweep, target_peak in CASES:
    p = Path(bundle_str)
    label = f"{p.parent.name}_{p.name}_sw{target_sweep}_pk{target_peak}"
    print(f"\n=== {p.name}  sweep={target_sweep}  peak={target_peak} ===")

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

    group   = df[df["sweep"] == target_sweep].sort_values("t_s")
    time    = group["t_s"].to_numpy()
    voltage = group["value"].to_numpy()

    sc_sw    = sweep_config.get("sweeps", {}).get(str(target_sweep), {})
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
    print(f"  Total peaks in sweep: {len(peaks)}")

    peak = int(peaks[target_peak])

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

    ks_pts = np.where(dvdt_up[thr_rel:] >= KINK_RATIO_THRESHOLD * max_dvdt)[0]
    ks_rel = thr_rel + int(ks_pts[0]) if len(ks_pts) > 0 else thr_rel

    kink_v    = v_up[ks_rel:up_rel + 1]
    kink_t    = t_up[ks_rel:up_rel + 1]
    kink_dvdt = dvdt_up[ks_rel:up_rel + 1]

    metrics = measure_kink_for_spike(kink_v, kink_t, kink_dvdt, debug=True)

    print(f"\n  --- Metrics ---")
    print(f"  num_kinks:        {metrics['num_kinks']}")
    print(f"  kink_ratio:       {metrics['kink_ratio']}")
    print(f"  kink_interval_ms: {metrics['kink_interval_ms']}")
    print(f"  kink_height_dvdt: {metrics['kink_height_dvdt']}")
    print(f"  kink_slope_dvdt:  {metrics['kink_slope_dvdt']}")

    if metrics['num_kinks'] == 0:
        print("  No kink detected — cannot verify slope.")
        continue

    kink_local = metrics['kink_idx']  # index within kink_v/kink_t/kink_dvdt

    # Re-derive the exact slope for verification
    t_rise  = (kink_t[0:kink_local + 1] - kink_t[0]) * 1000.0
    dv_rise = kink_dvdt[0:kink_local + 1]
    coeffs  = np.polyfit(t_rise, dv_rise, 1)
    slope_check = coeffs[0]
    print(f"\n  Slope re-derived manually: {slope_check:.4f} mV/ms²")
    print(f"  Matches metrics:           {np.isclose(slope_check, metrics['kink_slope_dvdt'])}")

    # --- Diagnostic plot ---
    t_rel = (kink_t - kink_t[0]) * 1000.0

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    # Voltage
    axes[0].plot(t_rel, kink_v, 'k-', lw=1.5, label='Voltage')
    axes[0].scatter(t_rel[0], kink_v[0], s=100, color='green', zorder=5, marker='o', label='Threshold')
    axes[0].scatter(t_rel[kink_local], kink_v[kink_local], s=120, color='orange', zorder=5, marker='s', label='Kink')
    axes[0].scatter(t_rel[-1], kink_v[-1], s=100, color='red', zorder=5, marker='^', label='Max upstroke')
    axes[0].set_ylabel('Voltage (mV)', fontsize=11, fontweight='bold')
    axes[0].set_title(f"{p.name}  sweep={target_sweep} peak={target_peak}\n"
                      f"slope={metrics['kink_slope_dvdt']:.2f} mV/ms²  |  "
                      f"ratio={metrics['kink_ratio']:.3f}  interval={metrics['kink_interval_ms']:.3f}ms",
                      fontsize=10, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # dV/dt
    axes[1].plot(t_rel, kink_dvdt, color='purple', lw=1.5, label='dV/dt')
    axes[1].scatter(t_rel[kink_local], kink_dvdt[kink_local], s=120, color='orange', zorder=5, marker='s')
    axes[1].scatter(t_rel[-1], kink_dvdt[-1], s=100, color='red', zorder=5, marker='^')

    # Overlay the line of best fit on the rising edge
    t_fit = t_rise
    fit_line = np.polyval(coeffs, t_fit)
    axes[1].plot(t_fit, fit_line, 'r--', lw=2, label=f'Slope fit: {slope_check:.1f} mV/ms²')
    axes[1].set_ylabel('dV/dt (mV/ms)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Time relative to threshold (ms)', fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / f"{p.name}_sw{target_sweep}_pk{target_peak}.jpeg"
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {out_path.name}")
