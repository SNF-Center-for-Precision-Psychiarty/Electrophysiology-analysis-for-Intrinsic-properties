"""
Merge kink_metrics_v2.csv into analysis.csv and AP_analysis.csv.

For every bundle under STUDY_DIR that has kink_metrics_v2.csv:
  analysis.csv     — updates: pct_spikes_with_kink, avg_kink_interval_ms,
                               avg_kink_ratio, avg_kink_height_dVdt
                     adds:    avg_kink_slope_dVdt  (after avg_kink_height_dVdt)
  AP_analysis.csv  — updates: Sweep_N_Kink_Count, Sweep_N_Kink_Interval_ms,
                               Sweep_N_Kink_Ratio, Sweep_N_Kink Height
                     adds:    Sweep_N_Kink_Slope   (after each Sweep_N_Kink Height)

Only sweeps present in kink_metrics_v2.csv are touched; others are left as-is.
"""

import numpy as np
import pandas as pd
from pathlib import Path

STUDY_DIR = r"Z:\Manos\SNF_Center Data - Manos\2. Electrophysiology\SNF_Center\Human_dataR"


def _to_num(series):
    """Convert a column that may contain empty strings to float."""
    return pd.to_numeric(series, errors="coerce")


def _update_analysis(p: Path, kink_df: pd.DataFrame, has_slope: bool):
    path = p / "analysis.csv"
    if not path.exists():
        return False

    df = pd.read_csv(path)

    for sweep in kink_df["sweep_number"].unique():
        sw       = kink_df[kink_df["sweep_number"] == sweep]
        detected = sw[sw["num_kinks"] > 0]
        total    = len(sw)

        pct          = 100.0 * len(detected) / total if total > 0 else np.nan
        avg_interval = detected["kink_interval_ms"].mean()  if len(detected) > 0 else np.nan
        avg_ratio    = detected["kink_ratio"].mean()         if len(detected) > 0 else np.nan
        avg_height   = detected["kink_height_dvdt"].mean()  if len(detected) > 0 else np.nan
        avg_slope    = detected["kink_slope_dvdt"].mean()   if has_slope and len(detected) > 0 else np.nan

        mask = df["sweep"] == sweep
        if not mask.any():
            continue
        df.loc[mask, "pct_spikes_with_kink"] = pct
        df.loc[mask, "avg_kink_interval_ms"] = avg_interval
        df.loc[mask, "avg_kink_ratio"]       = avg_ratio
        df.loc[mask, "avg_kink_height_dVdt"] = avg_height
        df.loc[mask, "avg_kink_slope_dVdt"]  = avg_slope

    # Reorder: insert avg_kink_slope_dVdt immediately after avg_kink_height_dVdt
    if "avg_kink_slope_dVdt" in df.columns and "avg_kink_height_dVdt" in df.columns:
        cols = list(df.columns)
        if cols.index("avg_kink_slope_dVdt") != cols.index("avg_kink_height_dVdt") + 1:
            cols.remove("avg_kink_slope_dVdt")
            ins = cols.index("avg_kink_height_dVdt") + 1
            cols.insert(ins, "avg_kink_slope_dVdt")
            df = df[cols]

    df.to_csv(path, index=False)
    return True


def _update_ap_analysis(p: Path, kink_df: pd.DataFrame, has_slope: bool):
    path = p / "AP_analysis.csv"
    if not path.exists():
        return False

    df = pd.read_csv(path)

    # Identify sweep numbers from existing Kink_Count columns
    sweep_nums = sorted(
        int(c.split("_")[1])
        for c in df.columns
        if c.startswith("Sweep_") and c.endswith("_Kink_Count")
    )

    processed_sweeps = set(kink_df["sweep_number"].unique())

    for sweep in sweep_nums:
        if sweep not in processed_sweeps:
            continue

        # Reset this sweep's kink columns to NaN before writing fresh values
        for col in [
            f"Sweep_{sweep}_Kink_Count",
            f"Sweep_{sweep}_Kink_Interval_ms",
            f"Sweep_{sweep}_Kink_Ratio",
            f"Sweep_{sweep}_Kink Height",
        ]:
            if col in df.columns:
                df[col] = np.nan
        if has_slope:
            df[f"Sweep_{sweep}_Kink_Slope"] = np.nan

        sw = kink_df[kink_df["sweep_number"] == sweep].set_index("spike_number")

        for spike_num, row in sw.iterrows():
            ap_idx   = spike_num - 1
            if ap_idx >= len(df):
                continue
            detected = bool(row["num_kinks"] > 0)

            df.at[ap_idx, f"Sweep_{sweep}_Kink_Count"]       = row["num_kinks"]
            df.at[ap_idx, f"Sweep_{sweep}_Kink_Interval_ms"] = row["kink_interval_ms"] if detected else np.nan
            df.at[ap_idx, f"Sweep_{sweep}_Kink_Ratio"]       = row["kink_ratio"]        if detected else np.nan
            df.at[ap_idx, f"Sweep_{sweep}_Kink Height"]      = row["kink_height_dvdt"]  if detected else np.nan
            if has_slope:
                df.at[ap_idx, f"Sweep_{sweep}_Kink_Slope"]   = row["kink_slope_dvdt"]   if detected else np.nan

    # Reorder: insert Sweep_N_Kink_Slope immediately after Sweep_N_Kink Height
    if has_slope:
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
        # Append any columns not yet placed (shouldn't happen, but safety net)
        for col in df.columns:
            if col not in added:
                new_cols.append(col)
        df = df[new_cols]

    df.to_csv(path, index=False)
    return True


def main():
    root    = Path(STUDY_DIR)
    bundles = sorted(csv.parent for csv in root.rglob("kink_metrics_v2.csv"))
    n       = len(bundles)
    print(f"Found {n} bundles with kink_metrics_v2.csv\n")

    n_ok = n_skip = n_err = 0

    for idx, p in enumerate(bundles, 1):
        print(f"[{idx:>3}/{n}] {p.name} ... ", end="", flush=True)
        try:
            kink_df   = pd.read_csv(p / "kink_metrics_v2.csv")
            has_slope = "kink_slope_dvdt" in kink_df.columns

            # Convert numeric columns (empty strings → NaN)
            for col in ["kink_ratio", "kink_interval_ms", "kink_height_dvdt"]:
                kink_df[col] = _to_num(kink_df[col])
            if has_slope:
                kink_df["kink_slope_dvdt"] = _to_num(kink_df["kink_slope_dvdt"])

            has_analysis = (p / "analysis.csv").exists()
            has_ap       = (p / "AP_analysis.csv").exists()

            if not has_analysis and not has_ap:
                print("skip (no analysis CSVs)")
                n_skip += 1
                continue

            parts = []
            if has_analysis and _update_analysis(p, kink_df, has_slope):
                parts.append("analysis.csv")
            if has_ap and _update_ap_analysis(p, kink_df, has_slope):
                parts.append("AP_analysis.csv")

            print(f"updated {', '.join(parts)}" + ("" if has_slope else "  [no slope — older kink_metrics_v2]"))
            n_ok += 1
        except Exception as e:
            import traceback
            print(f"ERROR: {e}")
            traceback.print_exc()
            n_err += 1

    print(f"\nDone. {n_ok} updated, {n_skip} skipped, {n_err} errors.")


if __name__ == "__main__":
    main()
