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
import shutil
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
    KINK_RATIO_THRESHOLD,
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
    "kink_slope_dvdt",
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

    # Clear any plots from a previous run so nothing stale remains
    if plot_dir.exists():
        shutil.rmtree(plot_dir)

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
            thr_rel = int(below[0])
            threshold_idx = w1_start + thr_rel

            if v_peak - float(v_up[thr_rel]) < MIN_PEAK_THRESHOLD_AMPLITUDE_MV:
                continue

            # Start kink search where dV/dt first reaches the ratio threshold —
            # nothing below this can ever pass the ratio gate, so skip it.
            ks_pts = np.where(dvdt_up[thr_rel:] >= KINK_RATIO_THRESHOLD * max_dvdt)[0]
            ks_rel = thr_rel + int(ks_pts[0]) if len(ks_pts) > 0 else thr_rel
            ks_idx = w1_start + ks_rel

            kink_v    = v_up[ks_rel:up_rel + 1]
            kink_t    = t_up[ks_rel:up_rel + 1]
            kink_dvdt = dvdt_up[ks_rel:up_rel + 1]

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
                    "kink_slope_dvdt":  "",
                })
                continue

            metrics = measure_kink_for_spike(kink_v, kink_t, kink_dvdt)

            rows.append({
                "bundle":           p.name,
                "sweep_number":     sweep_number,
                "spike_number":     i + 1,
                "stimulus_level_pA": stim_pA,
                "num_kinks":        metrics["num_kinks"],
                "kink_ratio":       metrics["kink_ratio"]       if metrics["num_kinks"] else "",
                "kink_interval_ms": metrics["kink_interval_ms"] if metrics["num_kinks"] else "",
                "kink_height_dvdt": metrics["kink_height_dvdt"] if metrics["num_kinks"] else "",
                "kink_slope_dvdt":  metrics["kink_slope_dvdt"]  if metrics["num_kinks"] else "",
            })

            if metrics["num_kinks"] > 0:
                kink_idx_global = ks_idx + metrics["kink_idx"]
                plot_kink_diagnostics(
                    voltage,
                    time,
                    threshold_idx,
                    kink_idx_global,
                    up_idx,
                    peak,
                    plot_dir,
                    f"sweep{sweep_number}_peak{i}",
                    search_start_idx=ks_idx,
                )

    return rows


def write_csv(p: Path, rows: list):
    csv_path = p / METRICS_FILE
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _to_num(val):
    """Convert a single value that may be an empty string to float."""
    if val == "" or val is None:
        return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def _update_analysis(p: Path, rows: list):
    """Write per-sweep kink aggregates into analysis.csv."""
    path = p / "analysis.csv"
    if not path.exists():
        return False

    df = pd.read_csv(path)

    # Wipe all kink columns before writing fresh values
    for col in ["pct_spikes_with_kink", "avg_kink_interval_ms", "avg_kink_ratio",
                "avg_kink_height_dVdt", "avg_kink_slope_dVdt"]:
        if col in df.columns:
            df[col] = np.nan

    # Group rows by sweep
    by_sweep = {}
    for r in rows:
        sw = r["sweep_number"]
        by_sweep.setdefault(sw, []).append(r)

    for sweep, spike_rows in by_sweep.items():
        detected = [r for r in spike_rows if r["num_kinks"]]
        total    = len(spike_rows)

        pct          = 100.0 * len(detected) / total if total > 0 else np.nan
        avg_interval = float(np.nanmean([_to_num(r["kink_interval_ms"]) for r in detected])) if detected else np.nan
        avg_ratio    = float(np.nanmean([_to_num(r["kink_ratio"])       for r in detected])) if detected else np.nan
        avg_height   = float(np.nanmean([_to_num(r["kink_height_dvdt"]) for r in detected])) if detected else np.nan
        avg_slope    = float(np.nanmean([_to_num(r["kink_slope_dvdt"])  for r in detected])) if detected else np.nan

        mask = df["sweep"] == sweep
        if not mask.any():
            continue
        df.loc[mask, "pct_spikes_with_kink"] = pct
        df.loc[mask, "avg_kink_interval_ms"] = avg_interval
        df.loc[mask, "avg_kink_ratio"]       = avg_ratio
        df.loc[mask, "avg_kink_height_dVdt"] = avg_height
        df.loc[mask, "avg_kink_slope_dVdt"]  = avg_slope

    # Ensure avg_kink_slope_dVdt sits immediately after avg_kink_height_dVdt
    if "avg_kink_slope_dVdt" in df.columns and "avg_kink_height_dVdt" in df.columns:
        cols = list(df.columns)
        if cols.index("avg_kink_slope_dVdt") != cols.index("avg_kink_height_dVdt") + 1:
            cols.remove("avg_kink_slope_dVdt")
            cols.insert(cols.index("avg_kink_height_dVdt") + 1, "avg_kink_slope_dVdt")
            df = df[cols]

    df.to_csv(path, index=False)
    return True


def _update_ap_analysis(p: Path, rows: list):
    """Write per-spike kink values into AP_analysis.csv."""
    path = p / "AP_analysis.csv"
    if not path.exists():
        return False

    df = pd.read_csv(path)

    sweep_nums = sorted(
        int(c.split("_")[1])
        for c in df.columns
        if c.startswith("Sweep_") and c.endswith("_Kink_Count")
    )

    # Wipe all kink columns before writing fresh values
    for col in df.columns:
        if "_Kink" in col:
            df[col] = np.nan

    by_sweep = {}
    for r in rows:
        by_sweep.setdefault(r["sweep_number"], []).append(r)

    for sweep in sweep_nums:
        if sweep not in by_sweep:
            continue

        for r in by_sweep[sweep]:
            ap_idx = r["spike_number"] - 1  # spike_number is 1-based
            if ap_idx >= len(df):
                continue
            detected = bool(r["num_kinks"])
            df.at[ap_idx, f"Sweep_{sweep}_Kink_Count"]       = r["num_kinks"]
            df.at[ap_idx, f"Sweep_{sweep}_Kink_Interval_ms"] = _to_num(r["kink_interval_ms"]) if detected else np.nan
            df.at[ap_idx, f"Sweep_{sweep}_Kink_Ratio"]       = _to_num(r["kink_ratio"])        if detected else np.nan
            df.at[ap_idx, f"Sweep_{sweep}_Kink Height"]      = _to_num(r["kink_height_dvdt"])  if detected else np.nan
            df.at[ap_idx, f"Sweep_{sweep}_Kink_Slope"]       = _to_num(r["kink_slope_dvdt"])   if detected else np.nan

    # Reorder: each Sweep_N_Kink_Slope immediately after its Sweep_N_Kink Height
    new_cols = []
    added    = set()
    for col in df.columns:
        new_cols.append(col)
        added.add(col)
        if col.endswith("_Kink Height"):
            slope_col = col.replace("_Kink Height", "_Kink_Slope")
            if slope_col in df.columns and slope_col not in added:
                new_cols.append(slope_col)
                added.add(slope_col)
    for col in df.columns:
        if col not in added:
            new_cols.append(col)
    df = df[new_cols]

    df.to_csv(path, index=False)
    return True


def prompt_study_dir() -> Path:
    while True:
        raw = input("Enter the root directory to scan for bundles:\n> ").strip().strip('"').strip("'")
        p = Path(raw)
        if p.is_dir():
            return p
        print(f"  Directory not found: {raw}\n  Please try again.\n")


def main():
    study_dir = prompt_study_dir()
    print(f"\nScanning: {study_dir}\n")
    bundles = find_bundles(study_dir)
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
            n_kinks  = sum(1 for r in rows if r["num_kinks"])
            n_spikes = len(rows)
            pct      = 100 * n_kinks / n_spikes if n_spikes else 0

            merged = []
            if _update_analysis(p, rows):
                merged.append("analysis.csv")
            if _update_ap_analysis(p, rows):
                merged.append("AP_analysis.csv")
            merge_note = f"  → merged into {', '.join(merged)}" if merged else ""

            print(f"{n_kinks}/{n_spikes} kinks ({pct:.0f}%){merge_note}")
            n_ok += 1
        except Exception as e:
            print(f"ERROR: {e}")
            n_err += 1

    print(f"\nDone. {n_ok} processed, {n_skip} skipped, {n_err} errors.")


if __name__ == "__main__":
    main()
