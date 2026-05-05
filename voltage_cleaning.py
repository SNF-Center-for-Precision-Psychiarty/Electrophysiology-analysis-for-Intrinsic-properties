"""
Canonical voltage-cleaning step for the analysis pipeline.

Applies a median filter to the stimulus-period voltage in negative-current sweeps,
eliminating spontaneous spikes that would contaminate downstream metrics
(sag, rheobase, kink detection, etc.).

The cleaned data is written to mV_*_clean.parquet and the bundle manifest is
updated so that manifest["tables"]["mv"] points to the cleaned file. The original
path is preserved as manifest["tables"]["mv_uncleaned"] for traceability.

This makes the cleaned file the single source of truth for all downstream readers
without requiring each module to be aware of the cleaning step.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import medfilt

# Median filter window in milliseconds. Wide enough to fully eliminate single APs
# (~2 ms wide) and short bursts. Narrow enough to preserve sag and other slow
# membrane features (which evolve over tens to hundreds of ms).
CLEAN_FILTER_WINDOW_MS = 4.0


def clean_voltage_for_negative_currents(bundle_dir):
    """
    Apply spike-removing median filter to negative-current sweeps in a bundle.

    Reads from the parquet file referenced by manifest["tables"]["mv_uncleaned"]
    if present, otherwise from manifest["tables"]["mv"]. This makes the function
    idempotent — re-running always cleans from the original, never compounding.

    Returns the path to the cleaned parquet.
    """
    bundle_path = Path(bundle_dir)
    manifest_path = bundle_path / "manifest.json"
    config_path = bundle_path / "sweep_config.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {bundle_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"sweep_config.json not found in {bundle_dir}")

    manifest = json.loads(manifest_path.read_text())
    config = json.loads(config_path.read_text())

    # Resolve the source file: original takes precedence so re-runs don't compound.
    tables = manifest.setdefault("tables", {})
    source_table = tables.get("mv_uncleaned") or tables["mv"]
    source_path = bundle_path / source_table
    df = pd.read_parquet(source_path)

    n_negative = 0
    n_cleaned = 0
    cleaned_chunks = []

    for sweep_id, group in df.groupby("sweep", sort=False):
        sweep_cfg = config.get("sweeps", {}).get(str(int(sweep_id)), {})
        stim_level = sweep_cfg.get("stimulus_level_pA")

        if stim_level is None or stim_level >= 0:
            cleaned_chunks.append(group)
            continue

        n_negative += 1

        windows = sweep_cfg.get("windows", {})
        stim_start = windows.get("stimulus_start_s")
        stim_end = windows.get("stimulus_end_s")
        if stim_start is None or stim_end is None:
            cleaned_chunks.append(group)
            continue

        t = group["t_s"].values
        v = group["value"].values
        if len(t) < 2:
            cleaned_chunks.append(group)
            continue

        dt = float(np.median(np.diff(t)))
        if dt <= 0:
            cleaned_chunks.append(group)
            continue

        window_samples = int(round(CLEAN_FILTER_WINDOW_MS / 1000.0 / dt))
        if window_samples < 3:
            window_samples = 3
        if window_samples % 2 == 0:
            window_samples += 1  # medfilt requires odd kernel

        stim_mask = (t >= stim_start) & (t <= stim_end)
        if stim_mask.sum() < window_samples:
            cleaned_chunks.append(group)
            continue

        v_clean = v.copy()
        v_clean[stim_mask] = medfilt(v[stim_mask], kernel_size=window_samples)

        new_group = group.copy()
        new_group["value"] = v_clean
        cleaned_chunks.append(new_group)
        n_cleaned += 1

    cleaned = pd.concat(cleaned_chunks, ignore_index=True)

    # Write alongside the original; name is derived from the original (not the
    # current "mv" pointer) so re-runs always overwrite the same _clean file.
    clean_path = source_path.with_name(source_path.stem + "_clean.parquet")
    cleaned.to_parquet(clean_path, index=False)

    # Preserve the original pointer once, then redirect "mv" to the clean file.
    if "mv_uncleaned" not in tables:
        tables["mv_uncleaned"] = source_table
    tables["mv"] = clean_path.name
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"  Median filter window: {CLEAN_FILTER_WINDOW_MS} ms")
    print(f"  Cleaned {n_cleaned}/{n_negative} negative-current sweeps")
    print(f"  Wrote: {clean_path.name}")
    print(f"  manifest['tables']['mv'] -> {clean_path.name}")

    return clean_path
