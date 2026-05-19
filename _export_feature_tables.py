"""
Export feature tracking tables as high-resolution PNG for PowerPoint insertion.

Outputs (saved next to this script):
  feature_table_detailed.png  — every metric, colour-coded by section
  feature_table_summary.png   — one row per section with feature count
"""

import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR  = Path(__file__).parent
FONT        = "Arial"
DPI         = 300

matplotlib.rcParams["font.family"] = FONT
matplotlib.rcParams["svg.fonttype"] = "none"

# ── Data ──────────────────────────────────────────────────────────────────────
SECTIONS = [
    {
        "name":    "Action Potential\nWaveform",
        "light":   "#D6EAF8",
        "dark":    "#2980B9",
        "summary": "Voltage landmarks and spike shape",
        "features": [
            ("Threshold voltage",           "Membrane potential at AP initiation (5% of max dV/dt)"),
            ("Peak voltage",                "Voltage at the apex of the spike"),
            ("Amplitude  (threshold -> peak)", "Voltage difference from threshold to peak"),
            ("Fast trough voltage",         "Voltage at end of repolarization (1% of min dV/dt)"),
            ("Spike height  (peak -> trough)", "Voltage drop from peak to fast trough"),
            ("AP width at half-height",     "Spike duration measured at the midpoint amplitude"),
        ],
    },
    {
        "name":    "Upstroke & Downstroke\nKinetics",
        "light":   "#D5F5E3",
        "dark":    "#27AE60",
        "summary": "Rates and timing of depolarization & repolarization",
        "features": [
            ("Max upstroke dV/dt",          "Peak rate of depolarization (mV/ms)"),
            ("Min downstroke dV/dt",        "Peak rate of repolarization (mV/ms, negative)"),
            ("Upstroke / downstroke ratio", "Relative strength of rise vs. fall"),
            ("Time: threshold -> peak",      "Total AP rise duration"),
            ("Time: max upstroke -> peak",   "Interval from fastest rise to voltage peak"),
            ("Voltage: max upstroke -> peak","Voltage gained after fastest rise point"),
            ("Time: peak -> max downstroke", "Onset of repolarization"),
            ("Time: peak -> fast trough",    "Total repolarization duration"),
            ("Time: threshold -> fast trough","Full AP duration"),
        ],
    },
    {
        "name":    "Firing\nProperties",
        "light":   "#E8DAEF",
        "dark":    "#8E44AD",
        "summary": "Spike train rate and regularity",
        "features": [
            ("Firing frequency",             "Spike count / stimulus duration (Hz)"),
            ("Mean ISI",                     "Average inter-spike interval (ms)"),
            ("ISI coefficient of variation", "Regularity of firing (SD / mean ISI)"),
            ("Binned spike count  (50 ms)",  "Instantaneous firing rate across stimulus"),
            ("Binned ISI CV  (50 ms)",       "Firing regularity across stimulus"),
        ],
    },
    {
        "name":    "Spike\nAdaptation",
        "light":   "#FDEBD0",
        "dark":    "#E67E22",
        "summary": "Changes in AP shape across the spike train",
        "features": [
            ("Width ratio  (middle / first)",       "Change in spike width over the train"),
            ("Width ratio  (last / first)",         ""),
            ("Amplitude ratio  (middle / first)",   "Change in spike height over the train"),
            ("Amplitude ratio  (last / first)",     ""),
            ("Fast trough ratio  (middle / first)", "Change in afterhyperpolarization depth"),
            ("Fast trough ratio  (last / first)",   ""),
        ],
    },
    {
        "name":    "Pre-Upstroke\nKink",
        "light":   "#FEF9E7",
        "dark":    "#D4AC0D",
        "summary": "Pre-upstroke inflection reflecting AIS–soma coupling",
        "features": [
            ("% spikes with kink",  "Prevalence of kinks across the sweep"),
            ("Kink interval (ms)",  "Time between kink and peak upstroke"),
            ("Kink ratio",          "dV/dt at kink / max upstroke dV/dt"),
            ("Kink height (dV/dt)", "Absolute dV/dt value at the kink point"),
            ("Kink slope (dV/dt²)", "Rate of dV/dt rise from search start to kink"),
        ],
    },
    {
        "name":    "Resting Membrane\nProperties",
        "light":   "#D1F2EB",
        "dark":    "#17A589",
        "summary": "Baseline cell state prior to stimulation",
        "features": [
            ("Resting membrane potential",
             "Average baseline voltage in the 100 ms pre-stimulus window"),
        ],
    },
]


# ── Drawing helpers ───────────────────────────────────────────────────────────
def _rect(ax, x, y, w, h, facecolor, edgecolor="white", lw=0.8):
    ax.add_patch(plt.Rectangle((x, y), w, h,
                                facecolor=facecolor, edgecolor=edgecolor,
                                linewidth=lw, clip_on=False))


def _text(ax, x, y, s, fontsize=9, color="#1a1a1a", bold=False,
          ha="left", va="center", wrap_width=None, style="normal"):
    if wrap_width and s:
        s = textwrap.fill(s, width=wrap_width)
    ax.text(x, y, s, fontsize=fontsize, color=color,
            fontweight="bold" if bold else "normal",
            fontstyle=style,
            ha=ha, va=va, multialignment=ha,
            clip_on=False)


# ── Detailed table ────────────────────────────────────────────────────────────
def draw_detailed_table(sections, path: Path):
    # Flatten all data rows, track section spans
    rows = []
    spans = []   # (y_bottom_in_rows, n_rows, section_dict)
    for sec in sections:
        spans.append((len(rows), len(sec["features"]), sec))
        for feat, desc in sec["features"]:
            rows.append((feat, desc, sec["light"]))

    n     = len(rows)
    HDR_H = 1.6   # header height in row-units
    ROW_H = 1.0
    TOTAL = HDR_H + n * ROW_H

    FIG_W, FIG_H = 16, TOTAL * 0.33
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, TOTAL)
    ax.axis("off")

    # Column layout  [section | feature | description]
    CX = [0.000, 0.175, 0.455]
    CW = [0.175, 0.280, 0.545]

    # ── Header ────────────────────────────────────────────────────────────────
    HDR_Y = n * ROW_H
    for label, x, w in zip(["Category", "Feature", "Description"], CX, CW):
        _rect(ax, x, HDR_Y, w, HDR_H, facecolor="#1C2833", lw=1.5)
        _text(ax, x + w / 2, HDR_Y + HDR_H / 2, label,
              fontsize=11, color="white", bold=True, ha="center")

    # ── Feature & description rows ────────────────────────────────────────────
    for i, (feat, desc, light) in enumerate(rows):
        y = (n - 1 - i) * ROW_H   # top row = index 0

        _rect(ax, CX[1], y, CW[1], ROW_H, facecolor=light)
        _text(ax, CX[1] + 0.008, y + ROW_H / 2, feat, fontsize=8.5)

        desc_bg = "#FFFFFF" if i % 2 == 0 else "#F4F6F7"
        _rect(ax, CX[2], y, CW[2], ROW_H, facecolor=desc_bg, edgecolor="#E8EAEB", lw=0.4)
        if desc:
            _text(ax, CX[2] + 0.008, y + ROW_H / 2, desc,
                  fontsize=8, color="#4A4A4A")

    # ── Section spanning cells (drawn last so they sit on top of row edges) ───
    for start_i, n_rows_sec, sec in spans:
        y_bottom = (n - start_i - n_rows_sec) * ROW_H
        span_h   = n_rows_sec * ROW_H
        _rect(ax, CX[0], y_bottom, CW[0], span_h,
              facecolor=sec["dark"], lw=1.2)
        _text(ax, CX[0] + CW[0] / 2, y_bottom + span_h / 2,
              sec["name"], fontsize=8.5, color="white", bold=True,
              ha="center", wrap_width=16)

    # Outer border
    _rect(ax, 0, 0, 1, TOTAL, facecolor="none", edgecolor="#AAB7B8", lw=2)

    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Summary table ─────────────────────────────────────────────────────────────
def draw_summary_table(sections, path: Path):
    n     = len(sections)
    HDR_H = 1.6
    ROW_H = 1.4
    TOTAL = HDR_H + n * ROW_H

    FIG_W, FIG_H = 13, TOTAL * 0.5
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, TOTAL)
    ax.axis("off")

    # Columns: [category | count | description]
    CX = [0.000, 0.420, 0.530]
    CW = [0.420, 0.110, 0.470]

    # ── Header ────────────────────────────────────────────────────────────────
    HDR_Y = n * ROW_H
    for label, x, w in zip(["Category", "Features", "What it captures"], CX, CW):
        _rect(ax, x, HDR_Y, w, HDR_H, facecolor="#1C2833", lw=1.5)
        _text(ax, x + w / 2, HDR_Y + HDR_H / 2, label,
              fontsize=11, color="white", bold=True, ha="center")

    # ── Section rows ──────────────────────────────────────────────────────────
    total_features = 0
    for i, sec in enumerate(sections):
        y    = (n - 1 - i) * ROW_H
        n_f  = len(sec["features"])
        total_features += n_f

        # Category cell (dark colour, white bold text)
        _rect(ax, CX[0], y, CW[0], ROW_H, facecolor=sec["dark"], lw=1.2)
        _text(ax, CX[0] + CW[0] / 2, y + ROW_H / 2,
              sec["name"].replace("\n", " "),
              fontsize=10, color="white", bold=True, ha="center")

        # Feature count (light colour, large bold number)
        _rect(ax, CX[1], y, CW[1], ROW_H, facecolor=sec["light"], lw=0.8)
        _text(ax, CX[1] + CW[1] / 2, y + ROW_H / 2, str(n_f),
              fontsize=15, color=sec["dark"], bold=True, ha="center")

        # Summary description
        desc_bg = "#FFFFFF" if i % 2 == 0 else "#F4F6F7"
        _rect(ax, CX[2], y, CW[2], ROW_H, facecolor=desc_bg, edgecolor="#E8EAEB", lw=0.4)
        _text(ax, CX[2] + 0.012, y + ROW_H / 2, sec["summary"],
              fontsize=9, color="#3D3D3D", style="italic")

    # Total row
    _rect(ax, 0, -ROW_H, 1, ROW_H, facecolor="#F0F3F4", edgecolor="#AAB7B8", lw=1.2)
    _text(ax, CX[0] + CW[0] / 2, -ROW_H / 2, "Total",
          fontsize=10, color="#1C2833", bold=True, ha="center")
    _text(ax, CX[1] + CW[1] / 2, -ROW_H / 2, str(total_features),
          fontsize=15, color="#1C2833", bold=True, ha="center")

    # Outer border
    _rect(ax, 0, -ROW_H, 1, TOTAL + ROW_H, facecolor="none", edgecolor="#AAB7B8", lw=2)

    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    draw_detailed_table(SECTIONS, OUTPUT_DIR / "feature_table_detailed.png")
    draw_summary_table(SECTIONS,  OUTPUT_DIR / "feature_table_summary.png")
    print("\nDone. Insert the PNG files directly into PowerPoint.")


if __name__ == "__main__":
    main()
