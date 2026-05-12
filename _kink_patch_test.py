import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _kink_patch_all import process_bundle, write_csv, METRICS_FILE, PLOT_SUBDIR

p = Path(r"Z:\Manos\SNF_Center Data - Manos\2. Electrophysiology\SNF_Center\Human_dataR\Developmental human study\sub-180216\sub-180216_ses-1802161oa-2-1-1_icephys")
rows = process_bundle(p)
write_csv(p, rows)
n_kinks = sum(1 for r in rows if r["num_kinks"])
print(f"{n_kinks}/{len(rows)} kinks detected")
print("CSV written:", (p / METRICS_FILE).exists())
plot_dir = p / PLOT_SUBDIR
plots = list(plot_dir.glob("*.jpeg")) if plot_dir.exists() else []
print(f"Plots generated: {len(plots)}")
for pl in sorted(plots):
    print(" ", pl.name)
