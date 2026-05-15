"""
Illustration-ready sweep plotter.

Prompts for a bundle directory and target current values (pA).
Finds the closest available sweeps, then plots voltage and current
traces side by side in a format suitable for Adobe Illustrator (SVG).

Annotations printed on the figure:
  - Average resting membrane potential (pre-stimulus baseline)
  - Actual current values that were plotted
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Plot parameters ────────────────────────────────────────────────────────────
PRE_PLOT_S       = 0.05    # seconds before stimulus start to show
POST_PLOT_S      = 0.15    # seconds after stimulus end to show
BASELINE_WIN_S   = 0.1     # seconds before stimulus used for RMP
LINE_WIDTH       = 1.5
FONT_FAMILY      = "Arial"
VOLTAGE_WIDTH_RATIO = 3    # voltage panel is 3x wider than current panel


# ── Input helpers ──────────────────────────────────────────────────────────────
def prompt_bundle() -> Path:
    while True:
        raw = input("Bundle directory:\n> ").strip().strip('"').strip("'")
        p = Path(raw)
        if p.is_dir() and (p / "manifest.json").exists():
            return p
        print(f"  Not a valid bundle (no manifest.json): {raw}\n")


def prompt_currents() -> list:
    while True:
        raw = input(
            "Target current values in pA (comma-separated, e.g. 50,100,200):\n> "
        ).strip()
        try:
            vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
            if vals:
                return vals
        except ValueError:
            pass
        print("  Could not parse — enter numbers separated by commas.\n")


# ── Data loading ───────────────────────────────────────────────────────────────
def _load_parquet(p: Path, prefix: str) -> pd.DataFrame:
    """Load clean > untagged > raw parquet for a given unit prefix."""
    for pattern in (f"{prefix}_*_clean.parquet", f"{prefix}_*.parquet"):
        hits = [f for f in p.rglob(pattern) if "_raw" not in f.name]
        if hits:
            return pd.read_parquet(hits[0]).sort_values(["sweep", "t_s"])
    raise FileNotFoundError(f"No {prefix} parquet found in {p}")


def load_data(p: Path):
    manifest     = json.loads((p / "manifest.json").read_text())
    sweep_config = json.loads((p / "sweep_config.json").read_text())
    voltage_df   = _load_parquet(p, "mV")

    try:
        current_df = _load_parquet(p, "pA")
    except FileNotFoundError:
        current_df = None
        print("  No pA parquet found — current panel will show stimulus step from config.")

    return voltage_df, current_df, sweep_config, manifest


# ── Sweep selection ────────────────────────────────────────────────────────────
def pick_sweeps(sweep_config: dict, targets: list) -> list:
    """
    For each target, return the valid sweep with the closest stimulus_level_pA.
    Returns list of dicts: {sweep_num, stimulus_pA, requested_pA}.
    """
    valid = {
        int(k): v
        for k, v in sweep_config.get("sweeps", {}).items()
        if v.get("valid", False) and v.get("stimulus_level_pA") is not None
    }
    if not valid:
        raise ValueError("No valid sweeps with stimulus_level_pA found in sweep_config.json")

    nums   = list(valid.keys())
    levels = np.array([valid[n]["stimulus_level_pA"] for n in nums])

    chosen = []
    seen   = set()
    for target in targets:
        idx       = int(np.argmin(np.abs(levels - target)))
        sn        = nums[idx]
        actual_pA = valid[sn]["stimulus_level_pA"]
        if sn in seen:
            print(f"  {target:+.0f} pA → sweep {sn} ({actual_pA:+.0f} pA) already selected, skipping duplicate")
            continue
        seen.add(sn)
        chosen.append({"sweep_num": sn, "stimulus_pA": actual_pA, "requested_pA": target})
        if abs(actual_pA - target) > 1e-3:
            print(f"  {target:+.0f} pA → closest is {actual_pA:+.0f} pA  (sweep {sn})")
        else:
            print(f"  {target:+.0f} pA → sweep {sn}")
    return chosen


# ── RMP ────────────────────────────────────────────────────────────────────────
def compute_rmp(voltage_df: pd.DataFrame, sweep_config: dict, sweep_nums: list) -> float:
    baselines = []
    sweeps_cfg = sweep_config.get("sweeps", {})
    for sn in sweep_nums:
        sc  = sweeps_cfg.get(str(sn), {})
        t0  = sc.get("windows", {}).get("stimulus_start_s")
        grp = voltage_df[voltage_df["sweep"] == sn]
        if t0 is None or grp.empty:
            continue
        mask = (grp["t_s"] >= t0 - BASELINE_WIN_S) & (grp["t_s"] < t0)
        if mask.any():
            baselines.append(float(grp.loc[mask, "value"].mean()))
    return float(np.mean(baselines)) if baselines else np.nan


# ── Figure ─────────────────────────────────────────────────────────────────────
def make_figure(
    bundle_path: Path,
    chosen: list,
    voltage_df: pd.DataFrame,
    current_df,
    sweep_config: dict,
    rmp: float,
) -> plt.Figure:

    matplotlib.rcParams["font.family"] = FONT_FAMILY
    matplotlib.rcParams["svg.fonttype"] = "none"   # keeps text editable in Illustrator

    n      = len(chosen)
    cmap   = matplotlib.colormaps.get_cmap("tab10" if n <= 10 else "tab20")
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    fig, (ax_v, ax_i) = plt.subplots(
        1, 2,
        figsize=(10, 4),
        gridspec_kw={"width_ratios": [VOLTAGE_WIDTH_RATIO, 1]},
    )

    # Determine shared time window from the first chosen sweep
    sc0          = sweep_config["sweeps"].get(str(chosen[0]["sweep_num"]), {})
    t_stim_start = sc0.get("windows", {}).get("stimulus_start_s", 0.0)
    t_stim_end   = sc0.get("windows", {}).get("stimulus_end_s",   1.0)
    t_min        = t_stim_start - PRE_PLOT_S
    t_max        = t_stim_end   + POST_PLOT_S

    actual_pAs = []

    for idx, info in enumerate(chosen):
        sn        = info["sweep_num"]
        color     = colors[idx]
        sc_sw     = sweep_config["sweeps"].get(str(sn), {})
        t0        = sc_sw.get("windows", {}).get("stimulus_start_s", t_stim_start)
        t_end     = sc_sw.get("windows", {}).get("stimulus_end_s",   t_stim_end)
        stim_pA   = info["stimulus_pA"]
        actual_pAs.append(stim_pA)
        label     = f"{stim_pA:+.0f} pA"

        # ── Voltage trace ──────────────────────────────────────────────────
        grp_v  = voltage_df[voltage_df["sweep"] == sn].sort_values("t_s")
        mask_v = (grp_v["t_s"] >= t_min) & (grp_v["t_s"] <= t_max)
        tv     = (grp_v.loc[mask_v, "t_s"].to_numpy() - t0) * 1000   # → ms, t=0 at stim start
        vv     = grp_v.loc[mask_v, "value"].to_numpy()
        ax_v.plot(tv, vv, color=color, linewidth=LINE_WIDTH, label=label)

        # ── Current trace ──────────────────────────────────────────────────
        if current_df is not None:
            grp_i  = current_df[current_df["sweep"] == sn].sort_values("t_s")
            mask_i = (grp_i["t_s"] >= t_min) & (grp_i["t_s"] <= t_max)
            ti     = (grp_i.loc[mask_i, "t_s"].to_numpy() - t0) * 1000
            iv     = grp_i.loc[mask_i, "value"].to_numpy()
        else:
            # Construct ideal step from config timing
            raw_t = grp_v["t_s"].to_numpy()
            mask_t = (raw_t >= t_min) & (raw_t <= t_max)
            ti = (raw_t[mask_t] - t0) * 1000
            iv = np.where(
                (raw_t[mask_t] >= t0) & (raw_t[mask_t] <= t_end),
                stim_pA, 0.0,
            )

        ax_i.plot(ti, iv, color=color, linewidth=LINE_WIDTH)

    # ── Axis styling ───────────────────────────────────────────────────────────
    for ax in (ax_v, ax_i):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", labelsize=9)
        ax.set_xlabel("Time from stimulus onset (ms)", fontsize=10)

    ax_v.set_ylabel("Membrane potential (mV)", fontsize=10)
    ax_i.set_ylabel("Current (pA)", fontsize=10)

    # ── Annotations ────────────────────────────────────────────────────────────
    rmp_str = f"RMP = {rmp:.1f} mV" if not np.isnan(rmp) else "RMP = n/a"
    ax_v.text(
        0.02, 0.02, rmp_str,
        transform=ax_v.transAxes,
        fontsize=9, va="bottom", ha="left",
    )

    pa_list = ", ".join(f"{v:+.0f}" for v in actual_pAs)
    fig.text(
        0.5, -0.04,
        f"Currents plotted: {pa_list} pA",
        ha="center", fontsize=9, color="dimgray",
    )

    # Legend inside voltage panel
    ax_v.legend(
        title="Stimulus",
        fontsize=8,
        title_fontsize=8,
        frameon=False,
        loc="upper right",
    )

    fig.suptitle(bundle_path.name, fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    p       = prompt_bundle()
    targets = prompt_currents()

    print(f"\nLoading data from: {p.name}")
    voltage_df, current_df, sweep_config, manifest = load_data(p)

    print("\nMatching sweeps:")
    chosen = pick_sweeps(sweep_config, targets)
    if not chosen:
        print("No sweeps selected — nothing to plot.")
        return

    sweep_nums = [c["sweep_num"] for c in chosen]
    rmp = compute_rmp(voltage_df, sweep_config, sweep_nums)
    if not np.isnan(rmp):
        print(f"\nRMP (avg over selected sweeps): {rmp:.2f} mV")

    print("\nBuilding figure...")
    fig = make_figure(p, chosen, voltage_df, current_df, sweep_config, rmp)

    out = p / "illustration_sweeps.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
