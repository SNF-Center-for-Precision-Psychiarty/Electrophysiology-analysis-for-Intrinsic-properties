"""
Pre-processing low-pass filter module

This module provides a butterworth low-pass filter implementation
to remove high-frequency noise from electrophysiology recordings
before analysis begins.

The filter operates on raw voltage and current data immediately
after loading, before any spike detection or analysis steps.
"""


import pandas as pd
import numpy as np
from scipy import signal
import json
from pathlib import Path
import warnings

# Suppress pandas FutureWarning about dtype compatibility during filtering
warnings.filterwarnings('ignore', category=FutureWarning, message='.*Setting an item of incompatible dtype.*')


def _contiguous_true_runs(mask: np.ndarray):
    """Yield (start, end) index pairs for contiguous True runs in a boolean mask."""
    if mask.size == 0:
        return
    edges = np.diff(mask.astype(np.int8))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1) + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size]
    for s, e in zip(starts, ends):
        yield int(s), int(e)


def apply_butterworth_lowpass(data_array: np.ndarray, 
                              sampling_rate: float,
                              cutoff_hz: float, 
                              order: int = 4) -> np.ndarray:
    """
    Apply a Butterworth low-pass filter to remove high-frequency noise.
    
    Args:
        data_array: Input signal (1D numpy array)
        sampling_rate: Sampling rate in Hz
        cutoff_hz: Cutoff frequency in Hz (REQUIRED parameter)
        order: Filter order (default 4)
        
    Returns:
        Filtered signal as numpy array
        
    Notes:
        - Butterworth filters have a flat passband (no ripple)
        - Cutoff frequency is -3dB point
        - Higher order = steeper rolloff but more phase distortion
        - Order 4 is a good compromise between steepness and stability
    """
    # Design the filter
    # Normalize frequency: cutoff_hz / (sampling_rate / 2)
    nyquist = sampling_rate / 2.0
    if cutoff_hz >= nyquist:
        raise ValueError(f"Cutoff frequency ({cutoff_hz} Hz) must be less than Nyquist frequency ({nyquist} Hz)")
    
    normalized_cutoff = cutoff_hz / nyquist
    b, a = signal.butter(order, normalized_cutoff, btype='low')
    
    # Handle NaNs robustly by filtering contiguous finite segments only.
    # Running filtfilt on arrays containing NaN can propagate NaNs widely.
    x = np.asarray(data_array, dtype=np.float64)
    filtered = x.copy()
    finite_mask = np.isfinite(x)

    # If no NaNs, fast path on whole array.
    if finite_mask.all():
        return signal.filtfilt(b, a, x)

    # filtfilt's default padding requirement.
    padlen = 3 * max(len(a), len(b))

    for start, end in _contiguous_true_runs(finite_mask):
        seg = x[start:end]
        # Segment too short for stable filtfilt: keep original values.
        if seg.size <= padlen:
            continue
        try:
            filtered[start:end] = signal.filtfilt(b, a, seg)
        except ValueError:
            # Be conservative: preserve original segment when filtering fails.
            continue

    return filtered.astype(data_array.dtype, copy=False)


def filter_sweep_data(df_sweep: pd.DataFrame, 
                      sampling_rate: float,
                      cutoff_hz: float) -> pd.DataFrame:
    """
    Apply low-pass filter to a single sweep's data.
    
    Args:
        df_sweep: DataFrame with columns like 'value' for one sweep
        sampling_rate: Sampling rate in Hz
        cutoff_hz: Cutoff frequency in Hz (REQUIRED parameter)
        
    Returns:
        DataFrame with filtered 'value' column
    """
    df_filtered = df_sweep.copy()
    
    if len(df_sweep) > 0:
        filtered_values = apply_butterworth_lowpass(
            df_sweep['value'].values,
            sampling_rate,
            cutoff_hz
        )
        df_filtered['value'] = filtered_values
    
    return df_filtered


def apply_lowpass_filter_to_bundle(bundle_dir: str, 
                                   cutoff_hz: float,
                                   inplace: bool = True,
                                   verbose: bool = False) -> dict:
    """
    Apply low-pass filter to voltage and current data in a bundle.
    
    This function reads the raw parquet files, applies a butterworth
    low-pass filter to remove high-frequency noise, and optionally
    overwrites the original files.
    
    Args:
        bundle_dir: Path to the bundle directory
        cutoff_hz: Cutoff frequency in Hz (REQUIRED parameter)
        inplace: If True, overwrites parquet files. If False, returns filtered data only.
        verbose: Print progress information
        
    Returns:
        Dictionary with:
        - 'df_mv': Filtered voltage dataframe
        - 'df_pa': Filtered current dataframe
        - 'cutoff_hz': Cutoff frequency used
        - 'n_sweeps_mv': Number of sweeps in voltage data
        - 'n_sweeps_pa': Number of sweeps in current data
        
    Notes:
        - Expects 'manifest.json' in bundle_dir
        - Modifies files in-place if inplace=True (recommended for pipeline)
        - Returns filtered data for inspection if inplace=False
    """
    p = Path(bundle_dir)
    
    # Load manifest to find parquet file paths
    manifest_path = p / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {bundle_dir}")
    
    man = json.loads(manifest_path.read_text())
    
    # Get sampling rates from manifest
    meta = man.get("meta", {})
    sample_rate_hz = meta.get("sampleRate_Hz")
    protocols = man.get("protocols", {})
    
    if sample_rate_hz is None:
        raise ValueError(f"sampleRate_Hz not found in manifest for {bundle_dir}")
    
    # Handle both single rate and list of rates
    if isinstance(sample_rate_hz, list):
        # Multiple rates - will use per-sweep rates from protocols dict
        # Build mapping from sweep ID to sampling rate
        sweep_to_rate = {}
        for protocol_id, protocol_info in protocols.items():
            rate_str = protocol_info.get("rate")
            if rate_str:
                sweep_to_rate[int(protocol_id)] = float(rate_str)
        
        if verbose:
            print(f"  Note: Multiple sampling rates detected")
            print(f"  Unique rates: {sorted(set(sweep_to_rate.values()))}")
            print(f"  Will use per-sweep rates from protocols")
    else:
        # Single rate - use for all sweeps
        fs_mv = float(sample_rate_hz)
        fs_pa = float(sample_rate_hz)
        sweep_to_rate = None
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"LOW-PASS FILTER PRE-PROCESSING")
        print(f"{'='*70}")
        print(f"Cutoff frequency: {cutoff_hz} Hz ({cutoff_hz/1000:.1f} kHz)")
        if sweep_to_rate is None:
            print(f"Sampling rate: {fs_mv} Hz")
        else:
            unique_rates = sorted(set(sweep_to_rate.values()))
            print(f"Sampling rates: {unique_rates} Hz (per-sweep)")
    
    # Load voltage data
    mv_path = p / man["tables"]["mv"]
    if verbose:
        print(f"\nLoading voltage data from {mv_path.name}...")
    df_mv = pd.read_parquet(mv_path)
    n_sweeps_mv = df_mv['sweep'].nunique()
    
    # Save backup of raw (unfiltered) voltage data BEFORE filtering
    mv_raw_path = p / man["tables"]["mv"].replace('.parquet', '_raw.parquet')
    df_mv.to_parquet(mv_raw_path, index=False)
    
    if verbose:
        print(f"  Loaded: {len(df_mv):,} samples, {n_sweeps_mv} sweeps")
        print(f"  Saved backup of raw data to {mv_raw_path.name}")
        print(f"  Filtering each sweep individually...")
    
    # Apply filter to each sweep separately
    df_mv_filtered = df_mv.copy()
    for sweep_id in sorted(df_mv['sweep'].unique()):
        mask = df_mv['sweep'] == sweep_id
        df_sweep = df_mv[mask]
        
        # Determine sampling rate for this sweep
        if sweep_to_rate is not None:
            # Multiple rates - get rate specific to this sweep
            fs_sweep = sweep_to_rate.get(int(sweep_id))
            if fs_sweep is None:
                raise ValueError(f"No sampling rate found in protocols for sweep {sweep_id}")
        else:
            # Single rate - use the same rate for all sweeps
            fs_sweep = fs_mv
        
        filtered_vals = apply_butterworth_lowpass(
            df_sweep['value'].values,
            fs_sweep,
            cutoff_hz
        )
        # Convert filtered values to match the column's original dtype (avoid pandas FutureWarning)
        original_dtype = df_mv_filtered['value'].dtype
        # Use .values assignment to bypass pandas dtype checking
        df_mv_filtered.loc[mask, 'value'] = pd.Series(
            filtered_vals.astype(original_dtype),
            index=df_sweep.index
        )
        
        
        if verbose and sweep_id % max(1, n_sweeps_mv // 10) == 0:
            print(f"    Progress: Sweep {sweep_id}/{n_sweeps_mv} (fs={fs_sweep} Hz)...")
    
    if verbose:
        print(f"  ✓ Voltage filtering complete")
    
    # Load current data
    pa_path = p / man["tables"]["pa"]
    if verbose:
        print(f"\nLoading current data from {pa_path.name}...")
    df_pa = pd.read_parquet(pa_path)
    n_sweeps_pa = df_pa['sweep'].nunique()
    
    # Save backup of raw (unfiltered) current data BEFORE filtering
    pa_raw_path = p / man["tables"]["pa"].replace('.parquet', '_raw.parquet')
    df_pa.to_parquet(pa_raw_path, index=False)
    
    if verbose:
        print(f"  Loaded: {len(df_pa):,} samples, {n_sweeps_pa} sweeps")
        print(f"  Saved backup of raw data to {pa_raw_path.name}")
        print(f"  Filtering each sweep individually...")
    
    # Apply filter to each sweep separately
    df_pa_filtered = df_pa.copy()
    for sweep_id in sorted(df_pa['sweep'].unique()):
        mask = df_pa['sweep'] == sweep_id
        df_sweep = df_pa[mask]
        
        # Determine sampling rate for this sweep
        if sweep_to_rate is not None:
            # Multiple rates - get rate specific to this sweep
            fs_sweep = sweep_to_rate.get(int(sweep_id))
            if fs_sweep is None:
                raise ValueError(f"No sampling rate found in protocols for sweep {sweep_id}")
        else:
            # Single rate - use the same rate for all sweeps
            fs_sweep = fs_pa
        
        filtered_vals = apply_butterworth_lowpass(
            df_sweep['value'].values,
            fs_sweep,
            cutoff_hz
        )
        # Convert filtered values to match the column's original dtype (avoid pandas FutureWarning)
        original_dtype = df_pa_filtered['value'].dtype
        # Use .values assignment to bypass pandas dtype checking
        df_pa_filtered.loc[mask, 'value'] = pd.Series(
            filtered_vals.astype(original_dtype),
            index=df_sweep.index
        )
        if verbose and sweep_id % max(1, n_sweeps_pa // 10) == 0:
            print(f"    Progress: Sweep {sweep_id}/{n_sweeps_pa} (fs={fs_sweep} Hz)...")
    
    if verbose:
        print(f"  ✓ Current filtering complete")
    
    # Save filtered data back to parquet if inplace=True
    if inplace:
        if verbose:
            print(f"\nSaving filtered data...")
        df_mv_filtered.to_parquet(mv_path, index=False)
        df_pa_filtered.to_parquet(pa_path, index=False)
        if verbose:
            print(f"  ✓ Filtered voltage saved to {mv_path.name}")
            print(f"  ✓ Filtered current saved to {pa_path.name}")
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"✓ LOW-PASS FILTER COMPLETE")
        print(f"{'='*70}\n")
    
    return {
        'df_mv': df_mv_filtered,
        'df_pa': df_pa_filtered,
        'cutoff_hz': cutoff_hz,
        'n_sweeps_mv': n_sweeps_mv,
        'n_sweeps_pa': n_sweeps_pa,
        'fs_mv': fs_mv,
        'fs_pa': fs_pa
    }


# if __name__ == "__main__":
#     # Example usage
#     import sys
    
#     if len(sys.argv) < 2:
#         print("Usage: python lowpass_filter.py <bundle_directory> [cutoff_hz]")
#         print("\nExample:")
#         print("  python lowpass_filter.py /path/to/bundle")
#         print("  python lowpass_filter.py /path/to/bundle 5000")
#         sys.exit(1)
    
#     bundle_dir = sys.argv[1]
#     cutoff_hz = float(sys.argv[2]) if len(sys.argv) > 2 else 5000
    
#     # Apply filter with verbose output
#     result = apply_lowpass_filter_to_bundle(bundle_dir, cutoff_hz, inplace=True, verbose=True)
#     print(f"Result: Filtered {result['n_sweeps_mv']} voltage sweeps and {result['n_sweeps_pa']} current sweeps")
