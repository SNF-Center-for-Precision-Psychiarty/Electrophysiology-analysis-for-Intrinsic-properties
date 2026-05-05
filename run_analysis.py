import json
from pathlib import Path
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from analysis import resting_vm_per_sweep, attach_manifest_to_analysis
from spike_detection_new import run_spike_detection
from sav_gol_filter import run_sav_gol
from input_resistance import get_input_resistance
from lowpass_filter import apply_lowpass_filter_to_bundle
from sag_current import calculate_sag_for_bundle
from matplotlib.backends.backend_pdf import PdfPages
import sys
import subprocess

def visualize_filter_all_sweeps(bundle_dir: str, max_sweeps: int = 4, cutoff_hz: Optional[int] = None, sampling_rate: Optional[float] = None):
    """
    Create before/after filter visualizations for all (or selected) sweeps in a bundle.
    
    Args:
        bundle_dir: Path to bundle directory
        max_sweeps: Maximum number of sweeps to visualize (default 12 for good coverage)
                   Set to None to visualize all sweeps
        cutoff_hz: Cutoff frequency in Hz - REQUIRED parameter
        sampling_rate: Sampling rate in Hz - used as fallback if manifest has single rate
    """   
    try:
        bundle_path = Path(bundle_dir)
        
        # Load manifest to get per-sweep sampling rates if mixed protocol
        manifest_path = bundle_path / "manifest.json"
        sweep_to_rate = {}
        
        if manifest_path.exists():
            man = json.loads(manifest_path.read_text())
            sample_rate_hz = man.get("meta", {}).get("sampleRate_Hz")
            protocols = man.get("protocols", {})
            
            if isinstance(sample_rate_hz, list):
                # Mixed protocol - build mapping from sweep ID to sampling rate
                for protocol_id, protocol_info in protocols.items():
                    rate_str = protocol_info.get("rate")
                    if rate_str:
                        sweep_to_rate[int(protocol_id)] = float(rate_str)
        
        # Get list of parquet files
        mv_files = list(bundle_path.rglob("mV_*.parquet"))
        if not mv_files:
            return
        
        # Count sweeps
        df = pd.read_parquet(mv_files[0])
        
        if 'sweep' in df.columns:
            n_sweeps = int(df['sweep'].max()) + 1
        else:
            n_sweeps = len(df.columns)
        
        # Determine how many to plot
        if max_sweeps is None:
            sweeps_to_plot = n_sweeps
        else:
            sweeps_to_plot = min(max_sweeps, n_sweeps)
        
        print(f"  Creating filter visualizations for {sweeps_to_plot} sweeps (of {n_sweeps} total)...")
        
        # Get current working directory to find plot script
        script_path = Path(__file__).parent / "plot_filter_before_after.py"
        if not script_path.exists():
            print(f"  ⚠ plot_filter_before_after.py not found, skipping visualizations")
            return
        
        for sweep_num in range(sweeps_to_plot):
            try:
                # Determine sampling rate for this sweep
                if sweep_to_rate:
                    # Mixed protocol - look up rate for this sweep
                    sweep_fs = sweep_to_rate.get(sweep_num, sampling_rate)
                else:
                    # Single protocol - use the provided rate
                    sweep_fs = sampling_rate
                
                # Run the visualization script
                cmd = [
                    "python",
                    str(script_path),
                    str(bundle_dir),
                    "--sweep", str(sweep_num),
                    "--cutoff", str(cutoff_hz),
                    "--sampling-rate", str(int(sweep_fs))
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    print(f"  ✓ Sweep {sweep_num}")
                else:
                    print(f"  ⚠ Sweep {sweep_num} failed: {result.stderr[:200]}")
            except subprocess.TimeoutExpired:
                print(f"  ⚠ Sweep {sweep_num} timed out")
            except Exception as e:
                print(f"  ⚠ Error with sweep {sweep_num}: {e}")
        
        # Print summary
        viz_dir = bundle_path / "filter_visualizations"
        if viz_dir.exists():
            n_plots = len(list(viz_dir.glob("*.jpeg")))
            print(f"  ✓ {n_plots} visualization files created")
            print(f"    Location: {viz_dir}")
        else:
            print(f"  ⚠ No visualizations directory found at {viz_dir}")
        
    except Exception as e:
        print(f"  ⚠ Could not generate filter visualizations: {e}")

def detect_hardware_malfunction(bundle_dir: str):
    """
    Detect if hardware malfunction occurred: both channels recorded as mV (empty pA).
    
    Args:
        bundle_dir: Path to the bundle
    
    Returns:
        True if malfunction detected (empty pA), False otherwise
    """
    p = Path(bundle_dir)
    man = json.loads((p / "manifest.json").read_text())
    
    try:
        df_pa = pd.read_parquet(p / man["tables"]["pa"])
        # Malfunction if pA is empty or has very few data points
        return len(df_pa) == 0 or df_pa.shape[0] < 100
    except:
        return False

def fix_hardware_malfunction_mV(bundle_dir: str):
    """
    When hardware malfunction occurs, two mV channels are recorded (correct + nonsense).
    This function identifies and keeps only the correct mV channel by checking signal stability.
    The correct channel should have consistent morphology across sweeps.
    The nonsense channel will have random/inconsistent data.
    
    Args:
        bundle_dir: Path to the bundle
    
    Returns:
        True if fix successful, False otherwise
    """
    p = Path(bundle_dir)
    man = json.loads((p / "manifest.json").read_text())
    
    try:
        mv_path = p / man["tables"]["mv"]
        df_mv = pd.read_parquet(mv_path)
        
        # Check if there are multiple channels
        if "channel_index" not in df_mv.columns:
            return False
        
        channels = df_mv["channel_index"].unique()
        if len(channels) != 2:
            return False
        
        print(f"  Detected {len(channels)} mV channels. Identifying the correct one...")
        
        # For each channel, calculate variance across sweeps
        # The correct channel should have consistent patterns (lower variance in peak detection)
        # The nonsense channel will have random data (higher variance)
        
        channel_stats = {}
        for ch in channels:
            df_ch = df_mv[df_mv["channel_index"] == ch]
            
            # Group by sweep and calculate signal statistics
            sweep_stats = df_ch.groupby("sweep")["value"].agg(["mean", "std", "min", "max", "count"])
            
            # Calculate coefficient of variation (std / mean) - indicator of signal consistency
            # Nonsense data will have very high CV
            cv_per_sweep = sweep_stats["std"] / (sweep_stats["mean"].abs() + 1e-6)
            avg_cv = cv_per_sweep.mean()
            
            channel_stats[ch] = {
                "avg_cv": avg_cv,
                "mean_std": sweep_stats["std"].mean(),
                "data_points": len(df_ch)
            }
            
            print(f"    Channel {ch}: CV={avg_cv:.4f}, Mean Std={sweep_stats['std'].mean():.4f}, Points={len(df_ch)}")
        
        # Select channel with HIGHER CV (more variable = correct channel with real signal)
        # The nonsense channel will have near-zero CV (flat noise or constant value)
        # The correct channel will have natural signal variation (higher CV)
        correct_channel = max(channel_stats.keys(), key=lambda x: channel_stats[x]["avg_cv"])
        
        print(f"  ✓ Selected Channel {correct_channel} as correct signal (highest CV)")
        
        # Keep only correct channel
        df_mv_fixed = df_mv[df_mv["channel_index"] == correct_channel].copy()
        
        # Save back
        df_mv_fixed.to_parquet(mv_path, index=False)
        print(f"  ✓ Saved corrected mV data to {mv_path}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ ERROR fixing mV data: {e}")
        return False

def is_current_data_valid(bundle_dir: str, sweep_config: Optional[dict] = None):
    """
    Check if current data exists in the expected stimulus time window.
    
    IMPORTANT: For MIXED PROTOCOL files only:
    sweep_config.json uses RELATIVE times per sweep (0-27s)
    but the parquet files use ABSOLUTE times (all sweeps concatenated, e.g., 278-1856s)
    This function converts relative times to absolute times for mixed protocol files.
    
    For SINGLE PROTOCOL files, times in sweep_config and parquet match directly.
    
    Args:
        bundle_dir: Path to the bundle
        sweep_config: Dict from sweep_config.json with stimulus windows (optional, tries to load if None)

    Returns:
        True if valid current data exists, False otherwise
    """
    p = Path(bundle_dir)
    man = json.loads((p / "manifest.json").read_text())
    df_pa = pd.read_parquet(p / man["tables"]["pa"])
    
    # Detect if mixed protocol
    is_mixed = "stimulus" in man["tables"] and "response" in man["tables"]
    
    # Determine time window from sweep_config or use first 10% of data
    if sweep_config is not None:
        try:
            first_valid_sweep_id = None
            t_min_relative = None
            t_max_relative = None
            
            for sweep_id_str, sweep_data in sweep_config.get("sweeps", {}).items():
                if sweep_data.get("valid", False):
                    first_valid_sweep_id = int(sweep_id_str)
                    t_min_relative = sweep_data["windows"].get("stimulus_start_s", 0.1)
                    t_max_relative = sweep_data["windows"].get("stimulus_end_s", 0.75)
                    break
            
            if first_valid_sweep_id is not None:
                # For mixed protocol: sweep_config contains ABSOLUTE times (from NWB file)
                # For single protocol: sweep_config contains RELATIVE times (within each sweep)
                if is_mixed:
                    # Mixed protocol: use absolute times directly from sweep_config
                    t_min = t_min_relative  # Actually absolute times, misnamed variable
                    t_max = t_max_relative  # Actually absolute times, misnamed variable
                else:
                    # Single protocol: use relative times directly
                    t_min = t_min_relative
                    t_max = t_max_relative
            else:
                # No valid sweeps found: use first 10% of data
                t_min = df_pa["t_s"].min()
                t_max = t_min + (df_pa["t_s"].max() - t_min) * 0.1
        except (KeyError, TypeError):
            # If sweep_config lookup fails, use first 10% of data
            t_min = df_pa["t_s"].min()
            t_max = t_min + (df_pa["t_s"].max() - t_min) * 0.1
    else:
        # No sweep_config: use first 10% of data for validation
        t_min = df_pa["t_s"].min()
        t_max = t_min + (df_pa["t_s"].max() - t_min) * 0.1
    
    df_pa_filtered = df_pa[(df_pa["t_s"] >= t_min) & (df_pa["t_s"] <= t_max)]
    return len(df_pa_filtered) > 0


def replace_current_data_with_reference(bundle_dir: str, reference_bundle_dir: str, sweep_config: Optional[dict] = None):
    """
    Replace the VALUES inside the faulty pA parquet file with values from a reference bundle.
    
    Crucially: The reference data sweep numbers are remapped to match the target bundle's sweep numbers,
    since both recordings use the same protocol but may have different sweep numbering.
    The TARGET FILENAME is preserved (e.g., pa_660.parquet stays pa_660.parquet).
    
    Args:
        bundle_dir: Path to the bundle with faulty current data (e.g., pa_660.parquet)
        reference_bundle_dir: Path to the reference bundle with good current data (e.g., pa_668.parquet)
    """
    p = Path(bundle_dir)
    p_ref = Path(reference_bundle_dir)
    
    # Load manifests
    man = json.loads((p / "manifest.json").read_text())
    man_ref = json.loads((p_ref / "manifest.json").read_text())
    
    # Get the pA parquet file paths
    pa_table_name = man["tables"]["pa"]  # e.g., "pa_660.parquet" (target filename to keep)
    pa_ref_table_name = man_ref["tables"]["pa"]  # e.g., "pa_668.parquet" (source)
    
    pa_ref_path = p_ref / pa_ref_table_name
    pa_path = p / pa_table_name  # Target path (keep this filename)
    
    # Load BOTH current datasets
    df_pa_faulty = pd.read_parquet(pa_path)  # Target (faulty) dataset
    df_pa_ref = pd.read_parquet(pa_ref_path)  # Source (reference) dataset
    
    # Get unique sweep numbers from each
    target_sweeps = sorted(df_pa_faulty["sweep"].unique())
    ref_sweeps = sorted(df_pa_ref["sweep"].unique())

    # If the faulty pA file has no sweeps (empty), we'll write the reference sweeps
    # into the target filename and use the reference sweep numbering.
    if len(target_sweeps) == 0:
        print("  Note: Faulty pA contains no sweeps. Will write reference sweeps into target file.")
        target_sweeps = list(ref_sweeps)

    print(f"  Target sweeps: {len(target_sweeps)} sweeps (e.g., {target_sweeps[:5]}...)")
    print(f"  Reference sweeps: {len(ref_sweeps)} sweeps (e.g., {ref_sweeps[:5]}...)")

    # Create mapping by position: map the first N reference sweeps to the first N target sweeps
    # If counts differ, map up to the smaller length and drop any unmapped reference rows.
    n_map = min(len(ref_sweeps), len(target_sweeps))
    if n_map == 0:
        raise ValueError("Reference or target pA has no sweeps to map")

    if len(ref_sweeps) != len(target_sweeps):
        print(f"  WARNING: Sweep count mismatch (ref={len(ref_sweeps)} vs target={len(target_sweeps)}). Mapping first {n_map} sweeps.")

    sweep_mapping = {ref_sweeps[i]: target_sweeps[i] for i in range(n_map)}

    # Remap reference data to target sweep numbers
    df_pa_ref_remapped = df_pa_ref.copy()
    df_pa_ref_remapped["sweep"] = df_pa_ref_remapped["sweep"].map(sweep_mapping)

    # Drop rows that could not be mapped (NaN sweep) to avoid NaN sweep ids
    before_rows = len(df_pa_ref_remapped)
    df_pa_ref_remapped = df_pa_ref_remapped.dropna(subset=["sweep"]).copy()
    after_rows = len(df_pa_ref_remapped)
    if after_rows < before_rows:
        print(f"  Note: Dropped {before_rows - after_rows} reference rows that could not be mapped to target sweeps.")

    # Ensure sweep is integer type
    df_pa_ref_remapped["sweep"] = df_pa_ref_remapped["sweep"].astype(int)
    # Preview summary and ask for confirmation before overwriting target file
    print("\n--- Preview replacement ---")
    print(f"Target (will keep filename): {pa_table_name} -> {pa_path}")
    print(f"Source (reference): {pa_ref_table_name} from {p_ref}")
    print(f"Remapped rows: {len(df_pa_ref_remapped)} (from {len(df_pa_ref)} source rows)")
    print(f"Target sweeps (post-map) sample: {sorted(df_pa_ref_remapped['sweep'].unique())[:8]}")
    print("First 5 rows of remapped reference data:")
    try:
        print(df_pa_ref_remapped.head().to_string())
    except Exception:
        print(df_pa_ref_remapped.head())

    # Apply baseline offset correction + per-sweep averaging + rounding to 5 pA increments
    try:
        # Step 1: Calculate baseline offset during quiet period (pre-stimulus period, no injection)
        # Use sweep_config if available, otherwise use first 10% of recording
        if sweep_config:
            try:
                # Find first sweep and get its stimulus start time
                for sweep_id, sweep_data in sweep_config.get("sweeps", {}).items():
                    if sweep_data.get("valid", False):
                        t_stim_start = sweep_data["windows"].get("stimulus_start_s", 0.1)
                        break
                baseline_window = df_pa_ref_remapped[df_pa_ref_remapped['t_s'] < t_stim_start]
                print(f"Using stimulus start time from sweep_config: {t_stim_start:.6f}s")
            except (KeyError, TypeError, StopIteration):
                # Fallback to first 10% if sweep_config extraction fails
                t_max = df_pa_ref_remapped['t_s'].max()
                baseline_window = df_pa_ref_remapped[df_pa_ref_remapped['t_s'] < (t_max * 0.1)]
                print(f"Using fallback: first 10% of recording (up to {t_max * 0.1:.6f}s)")
        else:
            # No sweep_config: use first 10% of recording as baseline
            t_max = df_pa_ref_remapped['t_s'].max()
            baseline_window = df_pa_ref_remapped[df_pa_ref_remapped['t_s'] < (t_max * 0.1)]
            print(f"No sweep_config provided: using first 10% of recording (up to {t_max * 0.1:.6f}s) as baseline")
        
        baseline_offset = baseline_window['value'].mean() if len(baseline_window) > 0 else 0.0
        print(f"\nBaseline offset (pre-stimulus quiet period): {baseline_offset:.2f} pA")

        # Step 2: Subtract baseline offset from all values
        df_pa_ref_remapped['value'] = df_pa_ref_remapped['value'] - baseline_offset

        # Step 3: Compute mean current in the stimulus window per sweep (after offset correction)
        # Again, use sweep_config if available
        if sweep_config:
            try:
                t_stim_start = None
                t_stim_end = None
                for sweep_id, sweep_data in sweep_config.get("sweeps", {}).items():
                    if sweep_data.get("valid", False):
                        windows = sweep_data["windows"]
                        t_stim_start = windows.get("stimulus_start_s", 0.1)
                        t_stim_end = windows.get("stimulus_end_s", 0.75)
                        break
                if t_stim_start is not None and t_stim_end is not None:
                    df_window = df_pa_ref_remapped[(df_pa_ref_remapped['t_s'] >= t_stim_start) & (df_pa_ref_remapped['t_s'] <= t_stim_end)]
                    print(f"Using stimulus window from sweep_config: [{t_stim_start:.6f}, {t_stim_end:.6f}]s")
                else:
                    raise KeyError("Could not extract stimulus window")
            except (KeyError, TypeError):
                # Fallback to 0.1-0.75 if extraction fails
                df_window = df_pa_ref_remapped[(df_pa_ref_remapped['t_s'] >= 0.1) & (df_pa_ref_remapped['t_s'] <= 0.75)]
                print("Using fallback stimulus window: [0.1, 0.75]s")
        else:
            # No sweep_config: use middle 50% of recording
            t_min = df_pa_ref_remapped['t_s'].min()
            t_max = df_pa_ref_remapped['t_s'].max()
            t_window_min = t_min + (t_max - t_min) * 0.2
            t_window_max = t_min + (t_max - t_min) * 0.7
            df_window = df_pa_ref_remapped[(df_pa_ref_remapped['t_s'] >= t_window_min) & (df_pa_ref_remapped['t_s'] <= t_window_max)]
            print(f"No sweep_config: using middle 50% of recording [{t_window_min:.6f}, {t_window_max:.6f}]s")
        
        avg_pa = df_window.groupby('sweep')['value'].mean().reset_index(name='avg_injected_current_pA')
        # if some sweeps missing in window, fallback to full-sweep mean
        if avg_pa['sweep'].nunique() < df_pa_ref_remapped['sweep'].nunique():
            fallback = df_pa_ref_remapped.groupby('sweep')['value'].mean().reset_index(name='avg_injected_current_pA')
            avg_pa = avg_pa.set_index('sweep').combine_first(fallback.set_index('sweep')).reset_index()

        # Step 4: Round to nearest 5 pA (or 0)
        avg_pa['avg_injected_current_pA_rounded'] = (np.round(avg_pa['avg_injected_current_pA'] / 5) * 5).astype(float)

        # Step 5: Apply rounded mean to all rows in each sweep
        for _, row in avg_pa.iterrows():
            sw = int(row['sweep'])
            rounded_val = float(row['avg_injected_current_pA_rounded'])
            df_pa_ref_remapped.loc[df_pa_ref_remapped['sweep'] == sw, 'value'] = rounded_val

        print('\nApplied baseline correction + per-sweep mean and rounded to 5 pA increments (preview):')
        print(avg_pa.head().to_string())
    except Exception as _e:
        print(f"Warning: could not apply per-sweep rounding to remapped data: {_e}")

    # Save remapped reference data to the TARGET filename, replacing the faulty values
    df_pa_ref_remapped.to_parquet(pa_path, index=False)
    print(f"✓ Replaced VALUES in {pa_table_name} (kept original filename)")
    print(f"  Source: {pa_ref_table_name} from {p_ref}")
    print(f"  Destination: {pa_path}")
    print(f"  Sweep remapping applied for {n_map} sweeps")


from sweep_classifier import classify_bundle_sweeps_nwb
from sweep_classifier import classify_bundle_sweeps_abf
from sweep_classifier import visualize_sweeps_from_parquet


def get_plot_preferences(no_checkpoints: bool = False) -> dict:
    """
    Ask user which supplemental plots to include in the analysis.
    
    Returns:
        dict: Plot preferences like {"kink_diagnostics": True, "sag_current": True, "savgol_plots": True}
    """
    
    # Check if running in interactive mode
    is_interactive = sys.stdin.isatty()
    
    if not is_interactive or no_checkpoints:
        # Default: include all supplemental plots in non-interactive mode
        print("\n[Auto] Including all supplemental plots (non-interactive mode)...")
        return {
            "kink_diagnostics": True,
            "sag_current": True,
            "savgol_plots": True
        }
    
    # Interactive mode: ask user
    print("\n" + "="*70)
    print("📊 SELECT SUPPLEMENTAL PLOTS FOR ANALYSIS")
    print("="*70)
    print("\nOptional supplemental plots that can be included in the final summary:")
    print("  1. Kink Diagnostics   - Detailed kink detection analysis per spike")
    print("  2. Sag Current        - HCN channel characterization plots")
    print("  3. SavGol Filter      - Savitzky-Golay filter analysis per sweep")
    print("\nEnter the plot numbers you want (i.e., '1,2' or '1' or '1,2,3'):")
    print("(Leave blank or press Enter for no supplemental plots): ")
    
    user_input = input().strip()
    
    # Default preferences (none included)
    preferences = {
        "kink_diagnostics": False,
        "sag_current": False,
        "savgol_plots": False
    }

    if user_input:
        # Parse user input
        try:
            selected = set()
            for item in user_input.split(','):
                selected.add(int(item.strip()))
            
            # Set preferences based on selection
            preferences["kink_diagnostics"] = 1 in selected
            preferences["sag_current"] = 2 in selected
            preferences["savgol_plots"] = 3 in selected
            
            # Print summary
            print("\n✓ Plot selection confirmed:")
            if preferences["kink_diagnostics"]:
                print("  ✓ Kink Diagnostics")
            if preferences["sag_current"]:
                print("  ✓ Sag Current")
            if preferences["savgol_plots"]:
                print("  ✓ SavGol Filter")
            if not any(preferences.values()):
                print("  (No supplemental plots selected)")
            print()
            
        except ValueError:
            print("\n⚠ Invalid input. No supplemental plots will be included.")
            print()
    else:
        print("\n✓ No supplemental plots selected (default).")
        print()

    return preferences


def generate_summary_plot(bundle_dir: str, plot_preferences: dict = None):
    """
    Collect all JPEG plot files from a bundle directory and combine them
    into a single master summary image.
    
    Gathers plots from (in order):
    - Sweep classification (sweeps_overlay.jpeg, all_sweeps_overview.jpeg)
    - AP_Per_Sweep/ (action potential per sweep)
    - Averaged_Peaks_Per_Sweep/ (averaged peaks)
    - SavGol_Plots/ (Savitzky-Golay filter) [optional]
    - InputResistance/ (input resistance analysis)
    - RMP distribution
    - Kink diagnostics [optional]
    - Sag current plots [optional]
    
    Args:
        bundle_dir: Path to bundle directory
        plot_preferences: Dict with keys 'kink_diagnostics', 'sag_current', 'savgol_plots' (boolean values)
                         If None, defaults to including all optional plots.
    """
    # Default preferences: include all optional plots if not specified
    if plot_preferences is None:
        plot_preferences = {
            "kink_diagnostics": False,
            "sag_current": False,
            "savgol_plots": False
        }
    
    try:
        from PIL import Image
    except ImportError:
        print("  WARNING: Pillow not installed. Skipping summary plot.")
        print("  Install with: pip install Pillow")
        return
    
    p = Path(bundle_dir)
    
    # Collect all image files in a structured order
    image_paths = []
    labels = []
    
    # 0. Sweep classification plots (created first during classification)
    sweeps_overlay = p / "sweeps_overlay.jpeg"
    if sweeps_overlay.exists():
        image_paths.append(sweeps_overlay)
        labels.append("Sweeps Overlay (All Sweeps)")
    
    all_sweeps_overview = p / "all_sweeps_overview.jpeg"
    if all_sweeps_overview.exists():
        image_paths.append(all_sweeps_overview)
        labels.append("All Sweeps Overview (Grid)")
    
    # 1. Individual AP plots
    ap_dir = p / "AP_Per_Sweep"
    if ap_dir.exists():
        for f in sorted(ap_dir.glob("AP_sweep_*.jpeg")):
            image_paths.append(f)
            sweep_num = f.stem.replace("AP_sweep_", "")
            labels.append(f"AP Sweep {sweep_num}")
    
    # 2. Individual Averaged Peaks plots
    avg_dir = p / "Averaged_Peaks_Per_Sweep"
    if avg_dir.exists():
        for f in sorted(avg_dir.glob("averaged_peaks_for_sweep_*.jpeg")):
            image_paths.append(f)
            sweep_num = f.stem.replace("averaged_peaks_for_sweep_", "")
            labels.append(f"Avg Peaks Sweep {sweep_num}")
    
    # 3. SavGol filter plots
    savgol_dir = p / "Sav_Gol_Plots_Per_Sweep"
    if savgol_dir.exists():
        for f in sorted(savgol_dir.glob("SavGol_Sweep*.jpeg")):
            image_paths.append(f)
            sweep_id = f.stem.replace("SavGol_Sweep", "").replace("_baseline", "")
            labels.append(f"SavGol Sweep {sweep_id}")
    
    # 4. RMP distribution post-filter
    rmp_plot = p / "RMP_Dist_Post_Filter.jpeg"
    if rmp_plot.exists():
        image_paths.append(rmp_plot)
        labels.append("RMP Distribution")
    
    # 5. Input Resistance
    ir_dir = p / "Input_Resistance"
    ir_plot = None
    if ir_dir.exists():
        ir_candidates = list(ir_dir.glob("InputResistance.jpeg"))
        if ir_candidates:
            ir_plot = ir_candidates[0]
    if ir_plot is None:
        # Check root of bundle
        ir_root = p / "InputResistance.jpeg"
        if ir_root.exists():
            ir_plot = ir_root
    if ir_plot:
        image_paths.append(ir_plot)
        labels.append("Input Resistance")
    
    # 6. Kept sweeps current
    kept_pa = p / "kept_sweeps_current.jpeg"
    if kept_pa.exists():
        image_paths.append(kept_pa)
        labels.append("Kept Sweeps - Current")
    
    # 7. Kept sweeps voltage
    kept_mv = p / "kept_sweeps_voltage.jpeg"
    if kept_mv.exists():
        image_paths.append(kept_mv)
        labels.append("Kept Sweeps - Voltage")
    
    # 8. Dropped sweeps current
    dropped_pa = p / "dropped_sweeps_current.jpeg"
    if dropped_pa.exists():
        image_paths.append(dropped_pa)
        labels.append("Dropped Sweeps - Current")

    # 9. Dropped sweeps voltage
    dropped_mv = p / "dropped_sweeps_voltage.jpeg"
    if dropped_mv.exists():
        image_paths.append(dropped_mv)
        labels.append("Dropped Sweeps - Voltage")
    
    # 10. Current grid (if exists)
    current_grid = p / "current_grid.jpeg"
    if current_grid.exists():
        image_paths.append(current_grid)
        labels.append("Current Grid (All Sweeps)")
    
    # 11. Voltage grid (if exists)
    voltage_grid = p / "voltage_grid.jpeg"
    if voltage_grid.exists():
        image_paths.append(voltage_grid)
        labels.append("Voltage Grid (All Sweeps)")
    
    # 12. Kink diagnostics
    kink_dir = p / "Kink_Diagnostics"
    if kink_dir.exists():
        for f in sorted(kink_dir.glob("*.jpeg")):
            image_paths.append(f)
            sweep_id = f.stem.replace("sweep_", "").replace("_kinks", "")
            labels.append(f"Kink Analysis Sweep {sweep_id}")
    
    if not image_paths:
        print("  No plot files found to combine.")
        return
    
    print(f"  Compiling all {len(image_paths)} plots...")
    
    pdf_path = p / "all_plots_summary.pdf"
    
    with PdfPages(pdf_path) as pdf:
        # PAGE 1: sweeps_overlay
        sweeps_overlay = p / "sweeps_overlay.jpeg"
        if sweeps_overlay.exists():
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111)
            img = Image.open(sweeps_overlay)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title("Sweeps Overlay", fontsize=14, fontweight='bold', pad=10)
            plt.tight_layout()
            pdf.savefig(fig, dpi=150, bbox_inches='tight')
            plt.close(fig)
            img.close()
        
        # PAGE 2: current_grid, voltage_grid (or stimulus_grid, response_grid for mixed protocol, or all_sweeps_overview)
        fig = plt.figure(figsize=(11, 11))
        current_grid = p / "current_grid.jpeg"
        voltage_grid = p / "voltage_grid.jpeg"
        stimulus_grid = p / "stimulus_grid.jpeg"
        response_grid = p / "response_grid.jpeg"
        all_sweeps_overview = p / "all_sweeps_overview.jpeg"
        
        if current_grid.exists() and voltage_grid.exists():
            ax1 = fig.add_subplot(2, 1, 1)
            img1 = Image.open(current_grid)
            ax1.imshow(img1)
            ax1.axis('off')
            ax1.set_title("Current Grid", fontsize=12, fontweight='bold')
            
            ax2 = fig.add_subplot(2, 1, 2)
            img2 = Image.open(voltage_grid)
            ax2.imshow(img2)
            ax2.axis('off')
            ax2.set_title("Voltage Grid", fontsize=12, fontweight='bold')
            img1.close()
            img2.close()
        elif stimulus_grid.exists() and response_grid.exists():
            ax1 = fig.add_subplot(2, 1, 1)
            img1 = Image.open(stimulus_grid)
            ax1.imshow(img1)
            ax1.axis('off')
            ax1.set_title("Stimulus Grid", fontsize=12, fontweight='bold')
            
            ax2 = fig.add_subplot(2, 1, 2)
            img2 = Image.open(response_grid)
            ax2.imshow(img2)
            ax2.axis('off')
            ax2.set_title("Response Grid", fontsize=12, fontweight='bold')
            img1.close()
            img2.close()
        elif all_sweeps_overview.exists():
            ax = fig.add_subplot(111)
            img = Image.open(all_sweeps_overview)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title("All Sweeps Overview", fontsize=12, fontweight='bold')
            img.close()
        
        if current_grid.exists() or voltage_grid.exists() or stimulus_grid.exists() or response_grid.exists() or all_sweeps_overview.exists():
            plt.tight_layout()
            pdf.savefig(fig, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # PAGE 3+: kept sweep summaries
        # If both single-protocol and mixed-protocol kept plots exist,
        # render them on separate pages to avoid subplot overlap.
        kept_pa = p / "kept_sweeps_current.jpeg"
        kept_mv = p / "kept_sweeps_voltage.jpeg"
        kept_stim = p / "kept_sweeps_stimulus.jpeg"
        kept_resp = p / "kept_sweeps_response.jpeg"

        def add_kept_sweeps_page(top_path, top_title, bottom_path, bottom_title, page_title):
            if not (top_path.exists() or bottom_path.exists()):
                return

            fig = plt.figure(figsize=(11, 11))

            if top_path.exists():
                ax1 = fig.add_subplot(2, 1, 1)
                img1 = Image.open(top_path)
                ax1.imshow(img1)
                ax1.axis('off')
                ax1.set_title(top_title, fontsize=12, fontweight='bold')
                img1.close()

            if bottom_path.exists():
                ax2 = fig.add_subplot(2, 1, 2)
                img2 = Image.open(bottom_path)
                ax2.imshow(img2)
                ax2.axis('off')
                ax2.set_title(bottom_title, fontsize=12, fontweight='bold')
                img2.close()

            fig.suptitle(page_title, fontsize=14, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig, dpi=150, bbox_inches='tight')
            plt.close(fig)

        has_single_kept = kept_pa.exists() or kept_mv.exists()
        has_mixed_kept = kept_stim.exists() or kept_resp.exists()

        if has_single_kept and has_mixed_kept:
            add_kept_sweeps_page(
                kept_pa,
                "Kept Sweeps - Current",
                kept_mv,
                "Kept Sweeps - Voltage",
                "Kept Sweeps (Current/Voltage)",
            )
            add_kept_sweeps_page(
                kept_stim,
                "Kept Sweeps - Stimulus",
                kept_resp,
                "Kept Sweeps - Response",
                "Kept Sweeps (Stimulus/Response)",
            )
        elif has_single_kept:
            add_kept_sweeps_page(
                kept_pa,
                "Kept Sweeps - Current",
                kept_mv,
                "Kept Sweeps - Voltage",
                "Kept Sweeps (Current/Voltage)",
            )
        elif has_mixed_kept:
            add_kept_sweeps_page(
                kept_stim,
                "Kept Sweeps - Stimulus",
                kept_resp,
                "Kept Sweeps - Response",
                "Kept Sweeps (Stimulus/Response)",
            )
        
        # PAGE 4: dropped_sweeps_current (or dropped_sweeps_stimulus for mixed protocol)
        dropped_pa = p / "dropped_sweeps_current.jpeg"
        dropped_stim = p / "dropped_sweeps_stimulus.jpeg"
        
        if (dropped_pa.exists() or dropped_stim.exists()):
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111)
            
            # Single protocol (current)
            if dropped_pa.exists():
                img = Image.open(dropped_pa)
                ax.imshow(img)
                ax.axis('off')
                ax.set_title("Dropped Sweeps - Current", fontsize=12, fontweight='bold')
                img.close()
            
            # Mixed protocol (stimulus)
            elif dropped_stim.exists():
                img = Image.open(dropped_stim)
                ax.imshow(img)
                ax.axis('off')
                ax.set_title("Dropped Sweeps - Stimulus", fontsize=12, fontweight='bold')
                img.close()
            
            plt.tight_layout()
            pdf.savefig(fig, dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        # PAGE 4b: dropped_sweeps_voltage (or dropped_sweeps_response for mixed protocol)
        dropped_mv = p / "dropped_sweeps_voltage.jpeg"
        dropped_resp = p / "dropped_sweeps_response.jpeg"
        
        if (dropped_mv.exists() or dropped_resp.exists()):
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111)
            
            # Single protocol (voltage)
            if dropped_mv.exists():
                img = Image.open(dropped_mv)
                ax.imshow(img)
                ax.axis('off')
                ax.set_title("Dropped Sweeps - Voltage", fontsize=12, fontweight='bold')
                img.close()
            
            # Mixed protocol (response)
            elif dropped_resp.exists():
                img = Image.open(dropped_resp)
                ax.imshow(img)
                ax.axis('off')
                ax.set_title("Dropped Sweeps - Response", fontsize=12, fontweight='bold')
                img.close()
            
            plt.tight_layout()
            pdf.savefig(fig, dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        # PAGE 5: AP_Per_Sweep in grid format
        ap_dir = p / "AP_Per_Sweep"
        if ap_dir.exists():
            ap_files = sorted(ap_dir.glob("AP_sweep_*.jpeg"))
            if ap_files:
                n_plots = len(ap_files)
                n_cols = 4
                n_rows = (n_plots + n_cols - 1) // n_cols
                
                fig = plt.figure(figsize=(14, 3.5 * n_rows))
                for idx, ap_file in enumerate(ap_files):
                    ax = fig.add_subplot(n_rows, n_cols, idx + 1)
                    img = Image.open(ap_file)
                    ax.imshow(img)
                    ax.axis('off')
                    sweep_num = ap_file.stem.replace("AP_sweep_", "")
                    ax.set_title(f"AP Sweep {sweep_num}", fontsize=10, fontweight='bold')
                    img.close()
                
                plt.tight_layout()
                pdf.savefig(fig, dpi=150, bbox_inches='tight')
                plt.close(fig)
        
        # PAGE 6: Averaged_Peaks_Per_Sweep in grid format
        avg_dir = p / "Averaged_Peaks_Per_Sweep"
        if avg_dir.exists():
            avg_files = sorted(avg_dir.glob("averaged_peaks_for_sweep_*.jpeg"))
            if avg_files:
                n_plots = len(avg_files)
                n_cols = 4
                n_rows = (n_plots + n_cols - 1) // n_cols
                
                fig = plt.figure(figsize=(14, 3.5 * n_rows))
                for idx, avg_file in enumerate(avg_files):
                    ax = fig.add_subplot(n_rows, n_cols, idx + 1)
                    img = Image.open(avg_file)
                    ax.imshow(img)
                    ax.axis('off')
                    sweep_num = avg_file.stem.replace("averaged_peaks_for_sweep_", "")
                    ax.set_title(f"Avg Peak Sweep {sweep_num}", fontsize=10, fontweight='bold')
                    img.close()
                
                plt.tight_layout()
                pdf.savefig(fig, dpi=150, bbox_inches='tight')
                plt.close(fig)
        
        # PAGE 7: Kink Diagnostics in grid format [OPTIONAL]
        if plot_preferences.get("kink_diagnostics", True):
            kink_dir = p / "Kink_Diagnostics"
            if kink_dir.exists():
                kink_files = sorted(kink_dir.glob("*.jpeg"))
                if kink_files:
                    n_plots = len(kink_files)
                    n_cols = 4
                    n_rows = (n_plots + n_cols - 1) // n_cols
                    
                    fig = plt.figure(figsize=(14, 3.5 * n_rows))
                    for idx, kink_file in enumerate(kink_files):
                        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
                        img = Image.open(kink_file)
                        ax.imshow(img)
                        ax.axis('off')
                        sweep_id = kink_file.stem.replace("sweep_", "").replace("_kinks", "")
                        ax.set_title(f"{sweep_id}", fontsize=10, fontweight='bold')
                        img.close()
                    
                    plt.tight_layout()
                    pdf.savefig(fig, dpi=150, bbox_inches='tight')
                    plt.close(fig)
        
        # PAGE 8: Sav_Gol_Plots_Per_Sweep in grid format and RMP plot [SavGol OPTIONAL]
        savgol_include = plot_preferences.get("savgol_plots", True)
        savgol_dir = p / "Sav_Gol_Plots_Per_Sweep"
        rmp_plot = p / "RMP_Dist_Post_Filter.jpeg"
        
        # Include RMP plot if SavGol is not selected (RMP is always included, just SavGol is optional)
        savgol_files = sorted(savgol_dir.glob("SavGol_Sweep*.jpeg")) if (savgol_include and savgol_dir.exists()) else []
        has_rmp = rmp_plot.exists()
        
        if savgol_files or has_rmp:
            n_savgol = len(savgol_files)
            
            # Calculate grid: 4 columns for SavGol plots, plus RMP if exists
            n_cols = 4
            n_savgol_rows = (n_savgol + n_cols - 1) // n_cols if n_savgol > 0 else 0
            n_total_rows = n_savgol_rows + (1 if has_rmp else 0)
            
            if n_total_rows > 0:
                fig = plt.figure(figsize=(14, 3.5 * n_total_rows))
                
                # Add SavGol plots (if included)
                for idx, savgol_file in enumerate(savgol_files):
                    ax = fig.add_subplot(n_total_rows, n_cols, idx + 1)
                    img = Image.open(savgol_file)
                    ax.imshow(img)
                    ax.axis('off')
                    sweep_id = savgol_file.stem.replace("SavGol_Sweep", "").replace("_baseline", "")
                    ax.set_title(f"SavGol Sweep {sweep_id}", fontsize=10, fontweight='bold')
                    img.close()
                
                # Add RMP plot if exists (always included regardless of SavGol preference)
                if has_rmp:
                    ax = fig.add_subplot(n_total_rows, n_cols, n_savgol + 1)
                    img = Image.open(rmp_plot)
                    ax.imshow(img)
                    ax.axis('off')
                    ax.set_title("RMP Distribution", fontsize=10, fontweight='bold')
                    img.close()
                
                plt.tight_layout()
                pdf.savefig(fig, dpi=150, bbox_inches='tight')
                plt.close(fig)
        
        # PAGE 9: Sag Current [OPTIONAL]
        if plot_preferences.get("sag_current", True):
            sag_dir = p / "SagCurrent"
            sag_files = sorted(sag_dir.glob("SagCurrent_sweep*.jpeg")) if sag_dir.exists() else []
            if sag_files:
                n_plots = len(sag_files)
                n_cols = min(4, n_plots)
                n_rows = (n_plots + n_cols - 1) // n_cols

                fig = plt.figure(figsize=(14, 3.5 * n_rows))
                for idx, sag_file in enumerate(sag_files):
                    ax = fig.add_subplot(n_rows, n_cols, idx + 1)
                    img = Image.open(sag_file)
                    ax.imshow(img)
                    ax.axis('off')
                    sweep_id = sag_file.stem.replace("SagCurrent_sweep", "")
                    ax.set_title(f"Sag Current Sweep {sweep_id}", fontsize=10, fontweight='bold')
                    img.close()

                fig.suptitle("Sag Current Analysis", fontsize=14, fontweight='bold')
                plt.tight_layout(rect=[0, 0, 1, 0.97])
                pdf.savefig(fig, dpi=150, bbox_inches='tight')
                plt.close(fig)
        
        # # PAGE 10: Filter Visualizations (before/after filtering plots)
        # filter_vis_dir = p / "filter_visualizations"
        # if filter_vis_dir.exists():
        #     filter_files = sorted(filter_vis_dir.glob("*.jpeg"))
        #     if filter_files:
        #         n_plots = len(filter_files)
        #         n_cols = 2  # 2 columns for filter visualizations (typically larger plots)
        #         n_rows = (n_plots + n_cols - 1) // n_cols
                
        #         fig = plt.figure(figsize=(14, 7 * n_rows))
        #         for idx, filter_file in enumerate(filter_files):
        #             ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        #             img = Image.open(filter_file)
        #             ax.imshow(img)
        #             ax.axis('off')
        #             # Extract a readable title from filename
        #             title = filter_file.stem.replace('_before_after_', ' - ').replace('_', ' ').replace('spectrum', 'Spectrum').replace('Sweep', 'Sweep')
        #             ax.set_title(title, fontsize=10, fontweight='bold')
        #             img.close()
                
        #         plt.tight_layout()
        #         pdf.savefig(fig, dpi=150, bbox_inches='tight')
        #         plt.close(fig)
        
        # PAGE 10: Input Resistance
        ir_dir = p / "Input_Resistance"
        ir_plot = None
        if ir_dir.exists():
            ir_candidates = list(ir_dir.glob("InputResistance.jpeg"))
            if ir_candidates:
                ir_plot = ir_candidates[0]
        if ir_plot is None:
            ir_root = p / "InputResistance.jpeg"
            if ir_root.exists():
                ir_plot = ir_root
        
        if ir_plot:
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111)
            img = Image.open(ir_plot)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title("Input Resistance", fontsize=14, fontweight='bold', pad=10)
            plt.tight_layout()
            pdf.savefig(fig, dpi=150, bbox_inches='tight')
            plt.close(fig)
            img.close()
    
    print(f"  ✓ Saved all plots summary: {pdf_path.name}")


def prompt_and_generate_sweep_gifs(bundle_dir: str, no_checkpoints: bool = False):
    """
    Ask the user if they want GIFs generated for the sweep grid plots (page 2 of the summary PDF).
    Generates GIFs from parquet data, cycling through sweeps one at a time.

    For single protocol: current (pA) and voltage (mV) GIFs
    For mixed protocol: stimulus and response GIFs

    Args:
        bundle_dir: Path to bundle directory
        no_checkpoints: If True, skip interactive prompt and generate GIFs automatically
    """
    from PIL import Image as PILImage
    import io

    p = Path(bundle_dir)
    man = json.loads((p / "manifest.json").read_text())
    is_mixed = "stimulus" in man.get("tables", {}) and "response" in man.get("tables", {})

    if is_mixed:
        label_top, label_bottom = "stimulus", "response"
    else:
        label_top, label_bottom = "current (pA)", "voltage (mV)"

    # Check interactive mode
    is_interactive = sys.stdin.isatty()

    if not is_interactive or no_checkpoints:
        print("\n[Auto] Skipping GIF generation (non-interactive mode).")
        return

    print(f"\n{'='*70}")
    print("🎞️  OPTIONAL: SWEEP GIF GENERATION")
    print("="*70)
    print(f"\nThe summary PDF (page 2) shows a grid of all sweeps.")
    print(f"Would you like to generate animated GIFs that cycle through each sweep?")
    print(f"  - {label_top} GIF (one frame per sweep)")
    print(f"  - {label_bottom} GIF (one frame per sweep)")
    print(f"\nEnter 'y' to generate GIFs, or press Enter to skip: ")

    user_input = input().strip().lower()
    if user_input not in ('y', 'yes'):
        print("✓ Skipping GIF generation.")
        return

    print(f"\nGenerating sweep GIFs...")

    # Load sweep config for stim window markers
    config_path = p / "sweep_config.json"
    stim_start, stim_end = None, None
    if config_path.exists():
        sweep_config = json.loads(config_path.read_text())
        stim_start = sweep_config.get("stim_start")
        stim_end = sweep_config.get("stim_end")

    if is_mixed:
        table_pairs = [
            ("stimulus", man["tables"]["stimulus"], "Stimulus", "value"),
            ("response", man["tables"]["response"], "Response", "value"),
        ]
    else:
        table_pairs = [
            ("current", man["tables"]["pa"], "Current (pA)", "pA"),
            ("voltage", man["tables"]["mv"], "Voltage (mV)", "mV"),
        ]

    for gif_name, table_file, title_prefix, ylabel in table_pairs:
        df = pd.read_parquet(p / table_file)
        sweeps = sorted(df["sweep"].unique())
        if len(sweeps) == 0:
            continue

        # Compute shared y-axis limits across all sweeps
        y_min = df["value"].min()
        y_max = df["value"].max()
        y_margin = (y_max - y_min) * 0.05
        y_min -= y_margin
        y_max += y_margin

        frames = []
        for sweep_id in sweeps:
            df_sweep = df[df["sweep"] == sweep_id]

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df_sweep["t_s"], df_sweep["value"], linewidth=0.8)
            if stim_start is not None:
                ax.axvline(stim_start, color='g', linestyle='--', alpha=0.7, label='Stim Start')
            if stim_end is not None:
                ax.axvline(stim_end, color='r', linestyle='--', alpha=0.7, label='Stim End')
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title_prefix} — Sweep {sweep_id}  ({sweeps.index(sweep_id)+1}/{len(sweeps)})")
            if stim_start is not None or stim_end is not None:
                ax.legend(loc='upper right')
            plt.tight_layout()

            # Render to PIL image
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=120)
            plt.close(fig)
            buf.seek(0)
            frame = PILImage.open(buf).copy()
            buf.close()
            frames.append(frame)

        if frames:
            gif_path = p / f"{gif_name}_sweeps.gif"
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=800,  # ms per frame
                loop=0
            )
            print(f"  ✓ Saved {gif_name} GIF ({len(frames)} frames): {gif_path.name}")

    print("✓ Sweep GIF generation complete.")

    # --- AP Per Sweep GIF ---
    ap_dir = p / "AP_Per_Sweep"
    ap_files = sorted(
        ap_dir.glob("AP_sweep_*.jpeg"),
        key=lambda f: int(f.stem.split("_")[-1]),
    ) if ap_dir.exists() else []
    if ap_files:
        print(f"\nFound {len(ap_files)} AP per-sweep plots in AP_Per_Sweep/.")
        print("Would you like to save these as an animated GIF?")
        print("Enter 'y' to generate, or press Enter to skip: ")
        user_input = input().strip().lower()
        if user_input in ('y', 'yes'):
            frames = [PILImage.open(f).copy() for f in ap_files]
            gif_path = p / "ap_per_sweep.gif"
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=800,
                loop=0
            )
            print(f"  ✓ Saved AP per-sweep GIF ({len(frames)} frames): {gif_path.name}")
        else:
            print("✓ Skipping AP per-sweep GIF.")

    # --- Averaged Peaks Per Sweep GIF ---
    avg_dir = p / "Averaged_Peaks_Per_Sweep"
    avg_files = sorted(
        avg_dir.glob("averaged_peaks_for_sweep_*.jpeg"),
        key=lambda f: int(f.stem.split("_")[-1]),
    ) if avg_dir.exists() else []
    if avg_files:
        print(f"\nFound {len(avg_files)} averaged peaks plots in Averaged_Peaks_Per_Sweep/.")
        print("Would you like to save these as an animated GIF?")
        print("Enter 'y' to generate, or press Enter to skip: ")
        user_input = input().strip().lower()
        if user_input in ('y', 'yes'):
            frames = [PILImage.open(f).copy() for f in avg_files]
            gif_path = p / "averaged_peaks_per_sweep.gif"
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=800,
                loop=0
            )
            print(f"  ✓ Saved averaged peaks GIF ({len(frames)} frames): {gif_path.name}")
        else:
            print("✓ Skipping averaged peaks GIF.")

    # --- Archive plot folders to save space ---
    plot_folders = [
        "AP_Per_Sweep",
        "Averaged_Peaks_Per_Sweep",
        "SagCurrent",
        "Sav_Gol_Plots_Per_Sweep",
        "Kink_Diagnostics",
        "Input_Resistance",
        # "Negative_Current_Smoothing",
        # "filter_visualizations",
    ]
    existing_folders = [p / name for name in plot_folders if (p / name).exists()]

    if existing_folders:
        folder_names = [f.name for f in existing_folders]
        total_files = sum(len(list(f.rglob("*"))) for f in existing_folders)
        print(f"\n{'='*70}")
        print("📦 OPTIONAL: ARCHIVE PLOT FOLDERS")
        print("="*70)
        print(f"\nFound {len(existing_folders)} plot folders ({total_files} files total):")
        for name in folder_names:
            print(f"  - {name}/")
        print(f"\nThese are already included in the summary PDF.")
        print("Would you like to zip them into plots_archive.zip and delete the originals?")
        print("Enter 'y' to archive, or press Enter to keep as-is: ")

        user_input = input().strip().lower()
        if user_input in ('y', 'yes'):
            import zipfile
            import shutil

            zip_path = p / "plots_archive.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for folder in existing_folders:
                    for file in sorted(folder.rglob("*")):
                        if file.is_file():
                            arcname = str(file.relative_to(p))
                            zf.write(file, arcname)

            # Verify zip was created successfully before deleting
            if zip_path.exists() and zip_path.stat().st_size > 0:
                for folder in existing_folders:
                    shutil.rmtree(folder)
                print(f"  ✓ Archived {len(existing_folders)} folders ({total_files} files) → {zip_path.name}")
                print(f"    Archive size: {zip_path.stat().st_size / 1024:.1f} KB")
            else:
                print("  ⚠ Zip creation failed. Original folders kept.")
        else:
            print("✓ Keeping plot folders as-is.")


def load_sweep_config(bundle_dir: str):
    """
    Load sweep_config.json if it exists, otherwise run sweep classification
    to detect current injection windows and generate one.
    
    For ABF bundles: Uses consistent window across all sweeps, keeps all sweeps
    For NWB bundles: Uses per-sweep window detection with validation
    
    Args:
        bundle_dir: Path to bundle directory
    
    Returns:
        dict: sweep_config (loaded or generated)
    """
    p = Path(bundle_dir)
    config_path = p / "sweep_config.json"
    
    # Check if this is an ABF bundle (has abf_path in manifest)
    manifest_path = p / "manifest.json"
    is_abf_bundle = False
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        is_abf_bundle = "abf_path" in manifest
    
    if config_path.exists():
        print(f"✓ Loading sweep_config.json from {p.name}")
        with open(config_path) as f:
            config = json.load(f)
        
        # Check if this was generated by the old crude auto-generation method
        # If so, re-run proper classification for accurate stimulus detection
        first_sweep = next(iter(config.get("sweeps", {}).values()), None)
        reason = first_sweep.get("reason", "") if first_sweep else ""
        reason = reason or ""  # Handle None value
        if first_sweep and "auto-generated" in reason:
            print(f"⚠ Found outdated auto-generated sweep_config in {p.name}")
            print("  Re-running sweep classifier for accurate stimulus window detection...")
            config_path.unlink()  # Delete old config
            
            # Use ABF-specific classifier for ABF bundles
            if is_abf_bundle:
                sweep_config = classify_bundle_sweeps_abf(bundle_dir, plot_sweeps=True)
            else:
                sweep_config = classify_bundle_sweeps_nwb(bundle_dir)
            
            if config_path.exists():
                with open(config_path) as f:
                    return json.load(f)
            return sweep_config
        
        # For ABF bundles, check if we need to regenerate with consistent window
        if is_abf_bundle and not config.get("consistent_window", False):
            print(f"⚠ ABF bundle has per-sweep windows, regenerating with consistent window...")
            config_path.unlink()
            sweep_config = classify_bundle_sweeps_abf(bundle_dir, plot_sweeps=True)
            if config_path.exists():
                with open(config_path) as f:
                    return json.load(f)
            return sweep_config
        
        return config
    else:
        print(f"⚠ No sweep_config.json found in {p.name}")
        print("  Running sweep classifier to detect current injection windows...")
        
        # Use ABF-specific classifier for ABF bundles
        if is_abf_bundle:
            sweep_config = classify_bundle_sweeps_abf(bundle_dir, plot_sweeps=True)
        else:
            sweep_config = classify_bundle_sweeps_nwb(bundle_dir)
        
        # Reload from the file that classify_bundle_sweeps_nwb wrote
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        return sweep_config


def run_for_bundle(bundle_dir: str, reference_bundle_dir: str = None, no_checkpoints: bool = False):
    p = Path(bundle_dir)
    pA_was_replaced = False  # Track if pA data was replaced
    
    print(f"\n{'='*70}")
    print(f"[Analysis] Starting analysis pipeline for bundle")
    print(f"{'='*70}")
    print(f"Bundle: {p.name}")
    
    # STEP 0a: Get user preferences for optional supplemental plots
    plot_preferences = get_plot_preferences(no_checkpoints=no_checkpoints)
    
    # STEP 0b: Load sweep_config early so we can use it for data processing
    sweep_config = load_sweep_config(bundle_dir)
    
    # STEP 1: Check for hardware malfunction (empty pA, 2 mV channels)
    if detect_hardware_malfunction(bundle_dir):
        print(f"\n⚠ HARDWARE MALFUNCTION DETECTED in {bundle_dir}")
        print("  Both channels recorded as voltage (empty current data).")
        
        # Step 1a: Fix mV data
        print("\n>>> Fixing voltage data: extracting correct mV channel...")
        if fix_hardware_malfunction_mV(bundle_dir):
            print("  ✓ Voltage data fixed")
        else:
            print("  ✗ Failed to fix voltage data")
        
        # Step 1b: Replace pA with reference
        print("\n>>> Replacing empty current data with reference recording...")
        
        if reference_bundle_dir is None:
            print("    No reference recording provided - skipping current data replacement")
            reference_bundle_dir = ""  # Auto-skip
        
        if reference_bundle_dir:
            try:
                replace_current_data_with_reference(bundle_dir, reference_bundle_dir, sweep_config)
                print("  ✓ Current data replaced")
                pA_was_replaced = True  # Mark that pA was replaced
            except Exception as e:
                print(f"  ✗ ERROR: Failed to load reference: {e}")
                print("    Proceeding with NaN current values...")
        else:
            print("    Proceeding without current data (input resistance analysis will have NaN)...")
        print()
    
    # STEP 2: Check for invalid current data (non-malfunction case)
    elif not is_current_data_valid(bundle_dir, sweep_config):
        print(f"\n⚠ WARNING: No valid current data found in {bundle_dir}")
        print("  Current data is required for accurate input resistance analysis.")
        
        # If no reference provided, auto-skip
        if reference_bundle_dir is None:
            print("\n>>> No reference recording provided - skipping current data replacement")
            reference_bundle_dir = ""  # Auto-skip
        
        if reference_bundle_dir:
            try:
                print(f"\n>>> Replacing faulty current data with reference recording...")
                replace_current_data_with_reference(bundle_dir, reference_bundle_dir, sweep_config)
                print()
            except Exception as e:
                print(f"✗ ERROR: Failed to load reference: {e}")
                print("  Proceeding with NaN current values...")
                print()
        else:
            print("  Proceeding without current data (results will have NaN for current-based metrics)...")
            print()
    else:
        print(f"\n✓ Current data looks valid in {bundle_dir} and no malfunction detected.\n")
    
    print(f"\n[Step 1] Loading data files...")
    man = json.loads((p / "manifest.json").read_text())

    # Ensure meta is a valid dict
    if not man.get("meta"):
        print(f"\n✗ ERROR: manifest.json in {bundle_dir} has no metadata.")
        print(f"  This bundle may need to be re-created. Skipping.")
        return

    # load tables
    print(f"  Loading voltage (mV) data...")
    df_mv = pd.read_parquet(p / man["tables"]["mv"])
    print(f"  ✓ Voltage: {df_mv.shape[0]:,} samples, {df_mv['sweep'].nunique()} sweeps")
    
    print(f"  Loading current (pA) data...")
    df_pa = pd.read_parquet(p / man["tables"]["pa"])
    print(f"  ✓ Current: {df_pa.shape[0]:,} samples, {df_pa['sweep'].nunique()} sweeps")
    
    # STEP 2: Apply low-pass filter based on sampling rate and generate sweep configuration
    print(f"\n[Step 2] Sweep configuration & filtering...")
    
    # Determine filter cutoff based on sampling rate
    sample_rate_hz = man["meta"].get("sampleRate_Hz")
    if isinstance(sample_rate_hz, list):
        # Multiple rates (mixed protocol) - use the maximum rate for Nyquist calculation
        fs = float(max(float(r) for r in sample_rate_hz))
    else:
        # Single rate
        fs = float(sample_rate_hz)
    nyquist_freq = fs / 2
    
    if fs >= 40000:
        # High sampling rate: apply 20 kHz low-pass filter
        filter_cutoff = 20000
        print(f"  High sampling rate ({fs} Hz) detected")
        print(f"  Applying {filter_cutoff/1000:.0f} kHz low-pass filter (pre-processing)...")
    else:
        # Low sampling rate: no filter (avoid over-filtering)
        filter_cutoff = None
        print(f"  Low sampling rate ({fs} Hz) detected")
        print(f"  Skipping low-pass filter (sampling rate below threshold)...")
    
    # Use kept_sweeps from sweep_config for low-pass filtering when available
    sweeps_to_filter = None
    if isinstance(sweep_config, dict):
        configured_kept = sweep_config.get("kept_sweeps", [])
        if configured_kept:
            sweeps_to_filter = configured_kept

    if filter_cutoff is not None and filter_cutoff < nyquist_freq:
        try:
            filter_result = apply_lowpass_filter_to_bundle(
                bundle_dir,
                cutoff_hz=filter_cutoff,
                sweeps_to_filter=sweeps_to_filter,
                inplace=True,
            )
            print(f"  ✓ Low-pass filter applied ({filter_cutoff/1000:.0f} kHz cutoff)")
            print(f"    - Filtered {filter_result['n_sweeps_mv']} voltage sweeps")
            print(f"    - Filtered {filter_result['n_sweeps_pa']} current sweeps")
            
            # Reload the filtered data
            df_mv = pd.read_parquet(p / man["tables"]["mv"])
            df_pa = pd.read_parquet(p / man["tables"]["pa"])
            
            # Generate filter visualizations only if filter was actually applied
            # print(f"  Generating filter visualizations...")
            # visualize_filter_all_sweeps(bundle_dir, cutoff_hz=filter_cutoff, sampling_rate=fs)
            print(f"  ✓ Sweep configuration & filtering complete")
        except Exception as e:
            print(f"  ⚠ WARNING: Low-pass filter failed: {e}")
            print(f"  Proceeding with unfiltered data...")
    else:
        print(f"  ℹ No filter applied (sampling rate {fs} Hz below threshold or invalid cutoff)")

    
    # Ensure sweep_config is a dict (None when no sweep_config.json exists)
    if sweep_config is None:
        sweep_config = {}

    # Determine kept/dropped sweeps once, then reuse for plotting and analysis
    kept_sweeps = sweep_config.get("kept_sweeps", [])
    dropped_sweeps = sweep_config.get("dropped_sweeps", [])

    # If no kept_sweeps defined, use all available sweeps
    if not kept_sweeps:
        kept_sweeps = sorted(df_mv["sweep"].unique().tolist())
        print(f"\n>>> No sweep filter defined - using all {len(kept_sweeps)} sweeps")
    else:
        print(f"\n>>> Filtering to kept sweeps: {len(kept_sweeps)} sweeps")

    # Generate kept/dropped sweep visualizations in Step 2 (configuration stage)
    print(f"  Generating kept/dropped sweep visualizations...")
    try:
        visualize_sweeps_from_parquet(bundle_dir, kept_sweeps, dropped_sweeps)
        print(f"  ✓ Kept/dropped sweep visualizations created")
    except Exception as e:
        print(f"  ⚠ WARNING: Failed to generate kept/dropped sweep visualizations: {e}")
    
    # Auto-skip the pause prompt for automated pipeline (no interactive input)
    # Check if we're running in non-interactive mode or with no_checkpoints flag
    is_interactive = sys.stdin.isatty()
    
    if is_interactive and not no_checkpoints:
        # Pause/resume loop for sweep config inspection (interactive mode only)
        while True:
            response = input("\nContinue to resting membrane potential calculation? (y/n): ").strip().lower()
            if response == 'y':
                break
            elif response == 'n':
                print(f"\n⏸ Pipeline paused. You can inspect files in:")
                print(f"  {bundle_dir}")
                print(f"\nWhen ready to resume, type 'resume':")
                resume_input = input().strip().lower()
                if resume_input == 'resume':
                    print("Resuming pipeline...")
                    continue
                else:
                    print("(Type 'resume' to continue)")
    else:
        # Auto-proceed in non-interactive mode
        print("\n[Auto] Proceeding with analysis (non-interactive mode)...")
    
    # Canonical voltage cleaning: remove spikes (incl. spontaneous) from negative-
    # current sweeps so all downstream metrics read from one consistent source.
    print(f"\n[Step 2.5] Cleaning voltage for negative-current sweeps...")
    try:
        from voltage_cleaning import clean_voltage_for_negative_currents
        clean_voltage_for_negative_currents(bundle_dir)
        # Reload manifest + voltage from the now-cleaned file
        man = json.loads((p / "manifest.json").read_text())
        df_mv = pd.read_parquet(p / man["tables"]["mv"])
        print(f"  Reloaded voltage from cleaned parquet ({df_mv.shape[0]:,} samples)")
    except Exception as e:
        print(f"  WARNING: Voltage cleaning failed: {e}")
        print(f"  Proceeding with un-cleaned voltage data...")

    print(f"\n[Step 3] Resting membrane potential calculation...")
    # Filter all dataframes to only include kept sweeps
    df_mv_kept = df_mv[df_mv["sweep"].isin(kept_sweeps)].copy()
    df_pa_kept = df_pa[df_pa["sweep"].isin(kept_sweeps)].copy()
    
    print(f"    mV data: {len(df_mv_kept)} rows (from {len(df_mv)})")
    print(f"    pA data: {len(df_pa_kept)} rows (from {len(df_pa)})")
    
    # sweep_config was already loaded at the beginning of this function
    print(f"  Calculating resting membrane potential...")
    df_vm_per_sweep = resting_vm_per_sweep(df_mv_kept, sweep_config, bundle_dir)  # one row per sweep, columns like resting_vm_mean_mV
    # Grand average resting vm
    combined_mean = float(df_vm_per_sweep["resting_vm_mean_mV"].mean())
    print(f"  ✓ Mean resting Vm: {combined_mean:.2f} mV")

    # save analysis outputs
    out_parq = p / "analysis.parquet"
    out_csv  = p / "analysis.csv"
    df_vm_per_sweep.to_parquet(out_parq, index=False)
    df_vm_per_sweep.to_csv(out_csv, index=False)

    # update manifest with analysis pointers (non-destructive)
    man.setdefault("analysis", {})
    man["analysis"]["resting_vm_table"] = out_parq.name
    man["analysis"]["resting_vm_mean"]  = combined_mean
    (p / "manifest.json").write_text(json.dumps(man, indent=2))

    print(f"\n✓ Resting Vm mean: {combined_mean:.2f} mV")
    print(f"  Saved to: {out_parq.name}")
    
    # Skip interactive pause in non-interactive mode or with no_checkpoints flag
    if is_interactive and not no_checkpoints:
        # Pause/resume loop for RMP inspection
        while True:
            response = input("\nContinue to spike detection? (y/n): ").strip().lower()
            if response == 'y':
                break
            elif response == 'n':
                print(f"\n⏸ Pipeline paused. You can inspect files in:")
                print(f"  {bundle_dir}")
                print(f"\nWhen ready to resume, type 'resume':")
                resume_input = input().strip().lower()
                if resume_input == 'resume':
                    print("Resuming pipeline...")
                    continue
                else:
                    print("(Type 'resume' to continue)")
        print(f"\n[Step 4] Spike detection")
    else:
        print(f"\n[Step 4] Spike detection")

    # spike detection
    print(f"  ⚡ Detecting action potentials...")
    # CRITICAL: Reload pA from disk to pick up replaced data (if malfunction was fixed above)
    df_pa_kept = pd.read_parquet(p / man["tables"]["pa"])
    df_pa_kept = df_pa_kept[df_pa_kept["sweep"].isin(kept_sweeps)].copy()
    df_analysis = pd.read_parquet(p /"analysis.parquet")
    fs = float(man["meta"]["sampleRate_Hz"])  # Convert from string to float

    run_spike_detection(df_mv_kept, df_pa_kept, df_analysis, fs, bundle_dir, 
                       pA_was_replaced=pA_was_replaced, sweep_config=sweep_config)
    #After running above line, analysis.parquet and analysis.csv and manifest.json will be updated
    
    print(f"  ✓ Spike detection complete")
    
    if is_interactive and not no_checkpoints:
        # Pause/resume loop for spike detection inspection
        while True:
            response = input("\nContinue to Savitzky-Goyal filtering? (y/n): ").strip().lower()
            if response == 'y':
                break
            elif response == 'n':
                print(f"\n⏸ Pipeline paused. You can inspect files in:")
                print(f"  {bundle_dir}")
                print(f"\nWhen ready to resume, type 'resume':")
                resume_input = input().strip().lower()
                if resume_input == 'resume':
                    print("Resuming pipeline...")
                    continue
                else:
                    print("(Type 'resume' to continue)")
        print(f"\n[Step 5] Savitzky-Golay filtering")
    else:
        print(f"\n[Step 5] Savitzky-Golay filtering")

    #low pass filter
    print(f"  🔄 Applying Savitzky-Golay low-pass filter...")
    df_analysis = pd.read_parquet(p /"analysis.parquet")
    run_sav_gol(df_mv_kept, df_analysis, fs, bundle_dir, sweep_config=sweep_config)
    
    print(f"  ✓ Savitzky-Golay filtering complete")
    
    if is_interactive and not no_checkpoints:
        # Pause/resume loop for SavGol inspection
        while True:
            response = input("\nContinue to input resistance calculation? (y/n): ").strip().lower()
            if response == 'y':
                break
            elif response == 'n':
                print(f"\n⏸ Pipeline paused. You can inspect files in:")
                print(f"  {bundle_dir}")
                print(f"\nWhen ready to resume, type 'resume':")
                resume_input = input().strip().lower()
                if resume_input == 'resume':
                    print("Resuming pipeline...")
                    continue
                else:
                    print("(Type 'resume' to continue)")
        print(f"\n[Step 6] Input resistance calculation")
    else:
        print(f"\n[Step 6] Input resistance calculation")
    
    #input resistance
    print(f"  ⚡ Computing input resistance...")
    # Reuse df_pa_kept from spike detection (pA data unchanged between steps)
    get_input_resistance(df_mv_kept, df_pa_kept, bundle_dir, sweep_config=sweep_config)
    #After running above line, manifest.json will be updated

    print(f"  ✓ Input resistance calculation complete")
    
    if is_interactive and not no_checkpoints:
        # Pause/resume loop for input resistance inspection
        while True:
            response = input("\nContinue to sag current analysis? (y/n): ").strip().lower()
            if response == 'y':
                break
            elif response == 'n':
                print(f"\n⏸ Pipeline paused. You can inspect files in:")
                print(f"  {bundle_dir}")
                print(f"\nWhen ready to resume, type 'resume':")
                resume_input = input().strip().lower()
                if resume_input == 'resume':
                    print("Resuming pipeline...")
                    continue
                else:
                    print("(Type 'resume' to continue)")

    # STEP 7: Sag current analysis (HCN channel characterization)
    print(f"\n[Step 7] Sag current analysis (HCN channels)...")
    print(f"  📊 Computing sag current from negative-current sweeps...")
    
    sag_results = calculate_sag_for_bundle(bundle_dir)
    
    if sag_results:
        # Add sag measurements to analysis.parquet
        df_analysis = pd.read_parquet(p / "analysis.parquet")

        # Initialize columns with NaN
        df_analysis['sag_voltage_mV'] = np.nan
        df_analysis['sag_ratio'] = np.nan
        df_analysis['sag_percent'] = np.nan
        df_analysis['peak_hyperpolarization_mV'] = np.nan

        # Fill in values for negative-current sweeps
        for sweep, measurements in sag_results['sag_results'].items():
            mask = df_analysis['sweep'] == sweep
            df_analysis.loc[mask, 'sag_voltage_mV'] = measurements['sag_voltage_mV']
            df_analysis.loc[mask, 'sag_ratio'] = measurements['sag_ratio']
            df_analysis.loc[mask, 'sag_percent'] = measurements['sag_percent']
            df_analysis.loc[mask, 'peak_hyperpolarization_mV'] = measurements['peak_hyperpolarization_mV']

        # Save updated analysis.parquet
        df_analysis.to_parquet(p / "analysis.parquet", index=False)

        print(f"  ✓ Sag current analysis complete")

        # STEP 7: After sag calculation
        print(f"\n{'='*70}")
        print("✓ STEP 7: SAG CURRENT ANALYSIS COMPLETE")
        print("="*70)
        if sag_results['summary']:
            print(f"Negative-current sweeps analyzed: {len(sag_results['hyper_sweeps'])}")
            print(f"Mean sag ratio: {sag_results['summary']['mean_sag_ratio']:.3f} ± {sag_results['summary']['std_sag_ratio']:.3f}")
        else:
            print(f"No negative-current sweeps found; sag columns were added as NaN values.")
        print(f"Sag columns added to analysis.parquet:")
        print(f"  - sag_voltage_mV: Absolute sag magnitude (mV)")
        print(f"  - sag_ratio: Sag as fraction of hyperpolarization (≈1.0 = complete recovery)")
        print(f"  - sag_percent: Sag as percentage")
        print(f"  - peak_hyperpolarization_mV: V_rest - V_min (mV)")
        print()
    else:
        print(f"  ⚠ Sag analysis could not run - missing required files")
    
    if is_interactive and not no_checkpoints:
        # Pause/resume loop for sag inspection
        while True:
            response = input("\nContinue to finalize results? (y/n): ").strip().lower()
            if response == 'y':
                break
            elif response == 'n':
                print(f"\n⏸ Pipeline paused. You can inspect files in:")
                print(f"  {bundle_dir}")
                print(f"\nWhen ready to resume, type 'resume':")
                resume_input = input().strip().lower()
                if resume_input == 'resume':
                    print("Resuming pipeline...")
                    continue
                else:
                    print("(Type 'resume' to continue)")

    print(f"\n{'='*70}")
    print("📊 Finalizing results...")
    print("="*70)

    #attach manifest details to analysis results
    df_analysis = pd.read_parquet(p /"analysis.parquet")
    attach_manifest_to_analysis(bundle_dir, df_analysis)
    print("Adding to analysis was successful")
    print(f"All updates completed and successful for {bundle_dir}.")

    # Generate master summary plot combining all figures
    print("🖼️  Generating summary PDF...")
    generate_summary_plot(bundle_dir, plot_preferences=plot_preferences)

    # Ask user if they want animated GIFs of sweep traces
    prompt_and_generate_sweep_gifs(bundle_dir, no_checkpoints=no_checkpoints)

    # FINAL CHECKPOINT: Pipeline complete
    print(f"\n{'='*70}")
    print("✅ ANALYSIS PIPELINE COMPLETE!")
    print("="*70)
    print(f"All analysis steps completed successfully for: {p.name}")
    print(f"Results saved to: {bundle_dir}")
    print(f"\n📁 Output files:")
    print(f"  - analysis.parquet: Complete results table")
    print(f"  - analysis.csv: Exported results (CSV format)")
    print(f"  - sweep_config.json: Sweep metadata and timing windows")
    print(f"  - Individual JPEG/PDF plots: In {bundle_dir}")
    print()

