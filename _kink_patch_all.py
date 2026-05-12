"""
Retroactive kink detection patch.

For every bundle under STUDY_DIR that has the required data files:
  - Re-runs kink detection with the unified algorithm
  - Saves diagnostic plots  → <bundle>/Kink_Diagnostics_v2/
  - Saves per-spike metrics → <bundle>/kink_metrics_v2.csv

Nothing existing is read from or written to.
"""

import sys
import json
import csv
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).parent))
from kink_detection import (
    measure_kink_for_spike,
    plot_kink_diagnostics,
    should_skip_kink_detection_struggling_cell,
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

STUDY_DIR = (
    r"Z:\Manos\SNF_Center Data - Manos\2. Electrophysiology\SNF_Center"
    r"\Human_dataR\Developmental human study"
)
PLOT_SUBDIR  = "Kink_Diagnostics_v2"
METRICS_FILE = "kink_metrics_v2.csv"

CSV_FIELDS = [
    "bundle",
    "sweep_number",
    "spike_number",
    "stimulus_level_pA",
    "num_kinks",
    "kink_ratio",
    "kink_interval_ms",
    "kink_height_dvdt",
]


def find_bundles(study_dir):
    root = Path(study_dir)
    bundles = []
    for manifest in sorted(root.rglob("manifest.json")):
        p = manifest.parent
        mv = list(p.rglob("mV_*_clean.parquet")) or list(p.rglob("mV_*.parquet"))
        sc = p / "sweep_config.json"
        if mv and sc.exists():
            bundles.append(p)
    return bundles


def process_bundle(p: Path):
    manifest   = json.loads((p / "manifest.json").read_text())
    fs_raw     = manifest.get("meta", {}).get("sampleRate_Hz", 50000)
    fs         = float(fs_raw) if not isinstance(fs_raw, list) else max(float(x) for x in fs_raw)

    mv_files   = list(p.rglob("mV_*_clean.parquet")) or list(p.rglob("mV_*.parquet"))
    df         = pd.read_parquet(mv_files[0]).sort_values(["sweep", "t_s"])
    sweep_config = json.loads((p / "sweep_config.json").read_text())

    # Reference stimulus window from first valid sweep
    t_stim_start = t_stim_end = None
    for sid, sdata in sweep_config.get("sweeps", {}).items():
        if sdata.get("valid", False):
            w = sdata.get("windows", {})
            t_stim_start = w.get("stimulus_start_s")
            t_stim_end   = w.get("stimulus_end_s")
            break
    if t_stim_start is None:
        return []   # no valid sweep; skip

    pre_s = PRE_THRESHOLD_WINDOW_S
    post_s = POST_THRESHOLD_WINDOW_S
    plot_dir = p / PLOT_SUBDIR
    rows = []

    for sweep_number in sorted(df["sweep"].unique()):
        group   = df[df["sweep"] == sweep_number].sort_values("t_s")
        time    = group["t_s"].to_numpy()
        voltage = group["value"].to_numpy()

        sc_sw    = sweep_config.get("sweeps", {}).get(str(sweep_number), {})
        sw_start = sc_sw["windows"]["stimulus_start_s"] if sc_sw and "windows" in sc_sw else t_stim_start
        sw_end   = sc_sw["windows"]["stimulus_end_s"]   if sc_sw and "windows" in sc_sw else t_stim_end
        stim_pA  = sc_sw.get("stimulus_level_pA", None)

        mask    = (time >= sw_start - pre_s) & (time <= sw_end + post_s)
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

        for i, peak in enumerate(peaks):
            peak   = int(peak)
            v_peak = float(voltage[peak])

            t_w1_start = max(time[peak] - pre_s, float(time[0]))
            w1_start   = int(np.searchsorted(time, t_w1_start, side="left"))
            w1_end     = peak + 1
            if w1_start >= w1_end:
                continue

            t_up    = time[w1_start:w1_end]
            v_up    = voltage[w1_start:w1_end]
            dvdt_up = np.gradient(v_up, t_up) * 1000
            if len(dvdt_up) == 0:
                continue

            max_dvdt = float(np.max(dvdt_up))
            up_rel   = int(np.argmax(dvdt_up))
            up_idx   = w1_start + up_rel

            thr_val = THRESHOLD_PERCENT * max_dvdt
            below   = np.where(dvdt_up >= thr_val)[0]
            if len(below) == 0:
                continue
            thr_rel      = int(below[0])
            threshold_idx = w1_start + thr_rel

            if v_peak - float(v_up[thr_rel]) < MIN_PEAK_THRESHOLD_AMPLITUDE_MV:
                continue

            kink_v    = v_up[thr_rel:up_rel + 1]
            kink_t    = t_up[thr_rel:up_rel + 1]
            kink_dvdt = dvdt_up[thr_rel:up_rel + 1]

            if should_skip_kink_detection_struggling_cell(spike_amplitudes, i):
                rows.append({
                    "bundle":           p.name,
                    "sweep_number":     sweep_number,
                    "spike_number":     i + 1,
                    "stimulus_level_pA": stim_pA,
                    "num_kinks":        0,
                    "kink_ratio":       "",
                    "kink_interval_ms": "",
                    "kink_height_dvdt": "",
                })
                continue

            metrics = measure_kink_for_spike(kink_v, kink_t, kink_dvdt)

            rows.append({
                "bundle":           p.name,
                "sweep_number":     sweep_number,
                "spike_number":     i + 1,
                "stimulus_level_pA": stim_pA,
                "num_kinks":        metrics["num_kinks"],
                "kink_ratio":       metrics["kink_ratio"] if metrics["num_kinks"] else "",
                "kink_interval_ms": metrics["kink_interval_ms"] if metrics["num_kinks"] else "",
                "kink_height_dvdt": metrics["kink_height_dvdt"] if metrics["num_kinks"] else "",
            })

            if metrics["num_kinks"] > 0:
                kink_idx_global = threshold_idx + metrics["kink_idx"]
                plot_kink_diagnostics(
                    voltage,
                    time,
                    threshold_idx,
                    kink_idx_global,
                    up_idx,
                    peak,           # correct peak index for this spike
                    plot_dir,
                    f"sweep{sweep_number}_peak{i}",
                )

    return rows


def write_csv(p: Path, rows: list):
    csv_path = p / METRICS_FILE
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    bundles = find_bundles(STUDY_DIR)
    n = len(bundles)
    print(f"Found {n} bundles to process\n")

    n_ok = n_err = n_skip = 0

    for idx, p in enumerate(bundles, 1):
        print(f"[{idx:>3}/{n}] {p.name} ... ", end="", flush=True)
        try:
            rows = process_bundle(p)
            if not rows:
                print("skipped (no valid sweeps)")
                n_skip += 1
                continue
            write_csv(p, rows)
            n_kinks = sum(1 for r in rows if r["num_kinks"])
            n_spikes = len(rows)
            pct = 100 * n_kinks / n_spikes if n_spikes else 0
            print(f"{n_kinks}/{n_spikes} kinks ({pct:.0f}%)")
            n_ok += 1
        except Exception as e:
            print(f"ERROR: {e}")
            n_err += 1

    print(f"\nDone. {n_ok} processed, {n_skip} skipped, {n_err} errors.")


if __name__ == "__main__":
    main()
