"""
Standalone kink detection verification — read-only, no files written.
Replicates the threshold/upstroke extraction from spike_detection_new.py
and calls measure_kink_for_spike directly.
"""

import sys
import json
import numpy as np
import pandas as pd
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

BUNDLES = [
    (
        r"Z:\Manos\SNF_Center Data - Manos\2. Electrophysiology\SNF_Center\Human_dataR"
        r"\Developmental human study\sub-180216\sub-180216_ses-1802161oa-2-1-1_icephys",
        [10, 11, 12, 13],
        False,
    ),
    (
        r"Z:\Manos\SNF_Center Data - Manos\2. Electrophysiology\SNF_Center\Human_dataR"
        r"\Developmental human study\sub-200123\sub-200123_ses-2001231tm-4-1-1_icephys",
        [22, 23, 24, 25, 26, 27, 28, 29],
        False,
    ),
    (
        r"Z:\Manos\SNF_Center Data - Manos\2. Electrophysiology\SNF_Center\Human_dataR"
        r"\Developmental human study\sub-200123\sub-200123_ses-2001231tm-3-1-1_icephys",
        None,
        False,
    ),
    (
        r"Z:\Manos\SNF_Center Data - Manos\2. Electrophysiology\SNF_Center\Human_dataR"
        r"\Developmental human study\sub-200123\sub-200123_ses-2001231tm-5-1-1_icephys",
        None,
        False,
    ),
    (
        r"Z:\Manos\SNF_Center Data - Manos\2. Electrophysiology\SNF_Center\Human_dataR"
        r"\Developmental human study\sub-200123\sub-200123_ses-2001231tm-5-3-2_icephys",
        None,
        True,   # verbose_misses: print gate values for missed spikes
    ),
    (
        r"Z:\Manos\SNF_Center Data - Manos\2. Electrophysiology\SNF_Center\Human_dataR"
        r"\Developmental human study\sub-150319\sub-150319_ses-1503191mg-1-2-1_icephys",
        [12, 13, 14, 15, 16],
        False,
    ),
]


def _gate_values(dvdt_window, times_window):
    """Compute the three gate values for a kink window without making a detection decision."""
    if len(dvdt_window) < 3:
        return None
    max_upstroke_idx    = int(np.argmax(dvdt_window))
    max_upstroke_height = dvdt_window[max_upstroke_idx]
    if max_upstroke_height <= 0:
        return None
    search = dvdt_window[0:max_upstroke_idx]
    if len(search) < 3:
        return {"note": f"search window only {len(search)} samples"}
    d2 = np.diff(search)
    if len(d2) < 3:
        return {"note": f"d2 only {len(d2)} samples"}
    valley_idx = None
    for i in range(1, len(d2) - 1):
        if d2[i] < d2[i - 1] and d2[i] < d2[i + 1]:
            if valley_idx is None or d2[i] < d2[valley_idx]:
                valley_idx = i
    if valley_idx is None:
        return {"note": "no valley in d2"}
    d2_left  = np.max(d2[:valley_idx])     if valley_idx > 0          else d2[0]
    d2_right = np.max(d2[valley_idx + 1:]) if valley_idx < len(d2) - 1 else d2[-1]
    d2_norm  = max(d2_left, d2_right)
    if d2_norm <= 0:
        return {"note": "d2_norm <= 0"}
    depth_ratio  = (d2_norm - d2[valley_idx]) / d2_norm
    kink_dvdt    = dvdt_window[valley_idx]
    kink_ratio   = kink_dvdt / max_upstroke_height
    interval_ms  = abs((times_window[valley_idx] - times_window[max_upstroke_idx]) * 1000.0)
    return {
        "kink_ratio": kink_ratio,
        "depth_ratio": depth_ratio,
        "interval_ms": interval_ms,
        "valley_idx": valley_idx,
        "window_n": len(search),
    }


def test_bundle(bundle_dir: str, target_sweeps, verbose_misses=False):
    p = Path(bundle_dir)
    print(f"\n{'='*70}")
    print(f"BUNDLE: {p.name}")

    manifest = json.loads((p / "manifest.json").read_text())
    fs_raw = manifest.get("meta", {}).get("sampleRate_Hz", 50000)
    fs = float(fs_raw) if not isinstance(fs_raw, list) else max(float(x) for x in fs_raw)

    mv_files = list(p.rglob("mV_*_clean.parquet")) or list(p.rglob("mV_*.parquet"))
    if not mv_files:
        print("  ERROR: no mV parquet found"); return
    df = pd.read_parquet(mv_files[0]).sort_values(["sweep", "t_s"])

    sc_path = p / "sweep_config.json"
    sweep_config = json.loads(sc_path.read_text()) if sc_path.exists() else {}

    t_stim_start, t_stim_end = None, None
    for sid, sdata in sweep_config.get("sweeps", {}).items():
        if sdata.get("valid", False):
            w = sdata.get("windows", {})
            t_stim_start = w.get("stimulus_start_s")
            t_stim_end   = w.get("stimulus_end_s")
            break
    if t_stim_start is None:
        print("  ERROR: no valid sweep in sweep_config"); return

    pre_s, post_s = PRE_THRESHOLD_WINDOW_S, POST_THRESHOLD_WINDOW_S
    bundle_total = bundle_kinks = 0

    for sweep_number in sorted(df["sweep"].unique()):
        if target_sweeps is not None and sweep_number not in target_sweeps:
            continue

        group = df[df["sweep"] == sweep_number].sort_values("t_s")
        time    = group["t_s"].to_numpy()
        voltage = group["value"].to_numpy()

        sc_sw = sweep_config.get("sweeps", {}).get(str(sweep_number), {})
        sw_start = sc_sw["windows"]["stimulus_start_s"] if sc_sw and "windows" in sc_sw else t_stim_start
        sw_end   = sc_sw["windows"]["stimulus_end_s"]   if sc_sw and "windows" in sc_sw else t_stim_end

        mask = (time >= sw_start - pre_s) & (time <= sw_end + post_s)
        time    = time[mask];  voltage = voltage[mask]
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

        n_spikes = n_kinks = 0
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
            if float(v_up[thr_rel]) - v_peak < -MIN_PEAK_THRESHOLD_AMPLITUDE_MV:
                pass  # amplitude ok (peak is higher than threshold)
            if v_peak - float(v_up[thr_rel]) < MIN_PEAK_THRESHOLD_AMPLITUDE_MV:
                continue

            kink_v    = v_up[thr_rel:up_rel + 1]
            kink_t    = t_up[thr_rel:up_rel + 1]
            kink_dvdt = dvdt_up[thr_rel:up_rel + 1]

            if should_skip_kink_detection_struggling_cell(spike_amplitudes, i):
                continue

            metrics = measure_kink_for_spike(kink_v, kink_t, kink_dvdt)
            n_spikes += 1
            if metrics["num_kinks"] > 0:
                n_kinks += 1
            elif verbose_misses:
                g = _gate_values(kink_dvdt, kink_t)
                if g is None or "note" in g:
                    print(f"    [MISS] sw{sweep_number} spike#{i+1}: {g}")
                else:
                    fails = []
                    if g["kink_ratio"]  < KINK_RATIO_THRESHOLD:     fails.append(f"ratio {g['kink_ratio']:.3f}<{KINK_RATIO_THRESHOLD}")
                    if g["depth_ratio"] < KINK_DEPTH_RATIO_THRESHOLD: fails.append(f"depth {g['depth_ratio']:.3f}<{KINK_DEPTH_RATIO_THRESHOLD}")
                    if g["interval_ms"] < KINK_MIN_INTERVAL_MS:       fails.append(f"interval {g['interval_ms']:.3f}ms<{KINK_MIN_INTERVAL_MS}ms")
                    print(f"    [MISS] sw{sweep_number} spike#{i+1}: n={g['window_n']} samples | FAIL: {', '.join(fails)}")

        if n_spikes > 0:
            pct  = 100 * n_kinks / n_spikes
            flag = "  OK" if pct > 50 else "  !!"
            print(f"{flag} Sweep {sweep_number:>3}: {n_kinks}/{n_spikes} spikes with kink  ({pct:.0f}%)")
            bundle_total += n_spikes;  bundle_kinks += n_kinks

    if bundle_total > 0:
        overall = 100 * bundle_kinks / bundle_total
        print(f"  --- Bundle total: {bundle_kinks}/{bundle_total} ({overall:.0f}%) ---")
    else:
        print("  (no valid spikes found in target sweeps)")


if __name__ == "__main__":
    for bundle_dir, target_sweeps, verbose_misses in BUNDLES:
        try:
            test_bundle(bundle_dir, target_sweeps, verbose_misses)
        except Exception as e:
            print(f"\nERROR in {Path(bundle_dir).name}: {e}")
            import traceback; traceback.print_exc()
    print("\nDone.")
