"""
Calculate sag current from hyperpolarizing current sweeps.

Sag is the voltage response during hyperpolarizing current injection,
caused by HCN (hyperpolarization-activated cyclic nucleotide-gated) channels.

Theory:
    When negative current is injected:
    1. Voltage initially hyperpolarizes (becomes more negative)
    2. Over time, HCN channels open, allowing positive current to flow back in
    3. Voltage "sags" or relaxes back toward less negative values
    4. The amount of sag indicates HCN channel activity

Measurements:
    - Sag voltage (mV): V_peak - V_steady
    - Sag ratio (dimensionless): (V_peak - V_steady) / (V_peak - V_rest)
        * Fraction of the hyperpolarization (rest → peak) that the cell recovered.
        * Typically positive and bounded in [0, 1] for normal sag.
    - Peak hyperpolarization (mV): V_rest - V_peak
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def find_negative_sweeps(analysis_parquet: pd.DataFrame, threshold_pA: float = 0):
    """
    Find all sweeps with injected current below the threshold.
    
    Returns:
        List of sweep numbers with injected current below threshold
    """

    negative_sweeps = analysis_parquet[
        analysis_parquet['avg_injected_current_pA'] < threshold_pA
    ]['sweep'].tolist()

    return [int(sweep) for sweep in negative_sweeps]


def measure_voltage_response(
    mv_data: pd.DataFrame,
    sweep: int,
    sweep_config: dict = None,
    sampling_rate: float = 200000  # Hz
) -> dict:
    """
    Measure key voltage points during a hyperpolarizing sweep.

    Updated definitions:
    - v_peak = most negative voltage in first 80 ms of stimulus
    - v_steady   = mean voltage in last 80 ms of stimulus
                   with 1 ms buffer before stimulus end
    """

    sweep_data = mv_data[mv_data['sweep'] == sweep]

    if len(sweep_data) == 0:
        return None

    # Remove NaN values (important for mixed protocol files)
    sweep_data = sweep_data[sweep_data['value'].notna()]
    
    if len(sweep_data) == 0:
        return None

    times = sweep_data['t_s'].values
    voltages = sweep_data['value'].values

    # Get stimulus window from config
    if sweep_config is None:
        sweep_config = {}

    sweep_str = str(int(sweep))

    if sweep_str in sweep_config:
        windows = sweep_config[sweep_str].get('windows', {})
        stimulus_start = windows.get('stimulus_start_s', 0.01)
        stimulus_end = windows.get('stimulus_end_s', times[-1])
    else:
        stimulus_start = 0.01
        stimulus_end = times[-1]

    # Extract stimulus portion
    stim_mask = (times >= stimulus_start) & (times <= stimulus_end)

    if stim_mask.sum() == 0:
        return None

    stim_times = times[stim_mask]
    stim_voltages = voltages[stim_mask]

    #v_peak
    peak_window_end = stimulus_start + 0.080

    peak_mask = (times >= stimulus_start) & (times <= peak_window_end)

    peak_voltages = voltages[peak_mask]

    if len(peak_voltages) > 0:
        v_peak = np.min(peak_voltages)
    else:
        v_peak = np.nan

    #v_steady
    # Last 80 ms of stimulus with 1 ms buffer before end
    steady_start = stimulus_end - 0.080 - 0.001  # 81ms before end = 80ms window + 1ms buffer
    steady_end = stimulus_end - 0.001            # 1ms before end (buffer)

    steady_mask = (times >= steady_start) & (times <= steady_end)

    steady_voltages = voltages[steady_mask]

    if len(steady_voltages) > 0:
        v_steady = np.mean(steady_voltages)
    else:
        v_steady = np.nan

    return {
        'v_peak': v_peak,
        'v_steady': v_steady,
    }


def calculate_sag(voltage_response: dict, v_rest: float) -> dict:
    """
    Calculate sag metrics from voltage response measurements.

    Args:
        voltage_response: Dict from measure_voltage_response()
        v_rest: Resting membrane potential (mV)

    Returns:
        Dictionary with:
        - 'sag_voltage_mV': Absolute sag (V_peak - V_steady, in mV)
        - 'sag_ratio': Fraction of hyperpolarization recovered:
                      (V_peak - V_steady) / (V_peak - V_rest)
                      Both numerator and denominator are negative under this convention,
                      yielding a positive ratio in [0, 1] for typical sag.
        - 'sag_percent': sag_ratio expressed as a percentage
    """
    if voltage_response is None:
        return None

    v_peak = voltage_response['v_peak']
    v_steady = voltage_response['v_steady']

    # Sag voltage (absolute sag, in mV)
    sag_voltage = v_peak - v_steady

    # Sag ratio: fraction of hyperpolarization (V_rest → V_peak) recovered by V_steady.
    # Numerator and denominator are both negative, so the ratio is positive.
    hyperpol_from_rest = v_peak - v_rest
    if hyperpol_from_rest != 0:
        sag_ratio = sag_voltage / hyperpol_from_rest
    else:
        sag_ratio = 0

    # Sag as percentage
    sag_percent = sag_ratio * 100

    return {
        'sag_voltage_mV': sag_voltage,
        'sag_ratio': sag_ratio,
        'sag_percent': sag_percent,
        'v_peak_mV': v_peak,
        'v_steady_mV': v_steady,
    }


def calculate_sag_for_bundle(
    bundle_dir: str, plot_preferences: dict = None) -> dict:
    """
    Calculate sag for all negative-current sweeps in a bundle.
    
    Args:
        bundle_dir: Path to bundle directory
        plot_preferences: Dict with keys 'kink_diagnostics', 'sag_current', 'savgol_plots' (boolean values)
                         If None, defaults to including all optional plots.
    
    Returns:
        Dictionary with results:
        - 'hyper_sweeps': List of all sweeps with injected current below 0 pA
        - 'sag_results': Dict mapping sweep_num → sag measurements
        - 'summary': Summary statistics
    """
    # Default plot preferences if not specified
    if plot_preferences is None:
        plot_preferences = {"kink_diagnostics": False, "sag_current": False, "savgol_plots": False}

    bundle_path = Path(bundle_dir)

    # -----------------------------
    # Locate required files via manifest (so we get the canonical cleaned mV file)
    # -----------------------------
    import json as _json
    manifest_path = bundle_path / "manifest.json"
    analysis_files = list(bundle_path.rglob("analysis.parquet"))
    sweep_config_files = list(bundle_path.rglob("sweep_config.json"))

    if not manifest_path.exists() or not analysis_files:
        print(f"⚠ Missing manifest or analysis parquet in {bundle_dir}")
        return None

    manifest = _json.loads(manifest_path.read_text())
    mv_table = manifest.get("tables", {}).get("mv")
    if not mv_table:
        print(f"⚠ No 'mv' entry in manifest for {bundle_dir}")
        return None

    mv_path = bundle_path / mv_table
    if not mv_path.exists():
        print(f"⚠ mV parquet not found at {mv_path}")
        return None

    # -----------------------------
    # Load data
    # -----------------------------
    mv_data = pd.read_parquet(mv_path)
    analysis_data = pd.read_parquet(analysis_files[0])

    # -----------------------------
    # Load sweep_config if available
    # -----------------------------
    sweep_config = {}

    if sweep_config_files:
        import json
        with open(sweep_config_files[0], 'r') as f:
            config_data = json.load(f)
            if 'sweeps' in config_data:
                sweep_config = config_data['sweeps']

    # -----------------------------
    # Identify all negative-current sweeps
    # -----------------------------
    hyper_sweeps = find_negative_sweeps(analysis_data, threshold_pA=0)

    print(f"\n[Sag Current Analysis]")
    print(f"  Using all negative-current sweeps (< 0 pA):")
    if hyper_sweeps:
        print(f"  Sweeps: {', '.join(str(sweep) for sweep in hyper_sweeps)}")
    else:
        print(f"  No sweeps found below 0 pA")

    # -----------------------------
    # Measure sag
    # -----------------------------
    sag_results = {}
    sag_ratios = []

    for sweep in hyper_sweeps:

        voltage_response = measure_voltage_response(
            mv_data,
            sweep,
            sweep_config=sweep_config
        )

        if voltage_response is None:
            continue

        sweep_row = analysis_data[analysis_data['sweep'] == sweep]
        resting_vm = sweep_row['resting_vm_mean_mV'].iloc[0]

        sag_measurements = calculate_sag(voltage_response, v_rest=resting_vm)

        peak_hyperpolarization = resting_vm - sag_measurements['v_peak_mV']

        sag_measurements['peak_hyperpolarization_mV'] = peak_hyperpolarization
        sag_measurements['v_rest_mV'] = resting_vm

        sag_results[sweep] = sag_measurements
        sag_ratios.append(sag_measurements['sag_ratio'])

        current = analysis_data[
            analysis_data['sweep'] == sweep
        ]['avg_injected_current_pA'].iloc[0]

        print(f"\n  Sweep {sweep} ({current:.0f} pA):")
        print(f"    V_peak: {sag_measurements['v_peak_mV']:.2f} mV")
        print(f"    V_rest:     {sag_measurements['v_rest_mV']:.2f} mV")
        print(f"    V_steady:   {sag_measurements['v_steady_mV']:.2f} mV")
        print(f"    Peak hyperpolarization: {sag_measurements['peak_hyperpolarization_mV']:.2f} mV")
        print(f"    Sag voltage: {sag_measurements['sag_voltage_mV']:.2f} mV")
        print(f"    Sag ratio:   {sag_measurements['sag_ratio']:.3f} ({sag_measurements['sag_percent']:.1f}%)")

        # -----------------------------
        # Generate diagnostic plot
        # -----------------------------

        sweep_data = mv_data[mv_data['sweep'] == sweep]

        times = sweep_data['t_s'].values
        voltages = sweep_data['value'].values

        sweep_str = str(int(sweep))

        if sweep_str in sweep_config:
            windows = sweep_config[sweep_str]['windows']
            stimulus_start = windows['stimulus_start_s']
            stimulus_end = windows['stimulus_end_s']
            baseline_start = windows.get('baseline_start_s', 0.0)
            baseline_end = windows.get('baseline_end_s', stimulus_start)
        else:
            stimulus_start = 0.01
            stimulus_end = times[-1]
            baseline_start = 0.0
            baseline_end = stimulus_start

        # Generate sag diagnostic plot [SAG PLOTS OPTIONAL]
        if plot_preferences.get("sag_current", True):
            plot_sag_diagnostics(
                bundle_dir,
                sweep,
                times,
                voltages,
                stimulus_start,
                stimulus_end,
                sag_measurements['v_peak_mV'],
                sag_measurements['v_steady_mV'],
                resting_vm,
                baseline_start,
                baseline_end,
            )

    # -----------------------------
    # Summary statistics
    # -----------------------------
    if sag_ratios:

        mean_sag = np.mean(sag_ratios)
        std_sag = np.std(sag_ratios)

        summary = {
            'n_sweeps': len(sag_results),
            'mean_sag_ratio': mean_sag,
            'std_sag_ratio': std_sag,
            'min_sag_ratio': np.min(sag_ratios),
            'max_sag_ratio': np.max(sag_ratios),
        }

    else:
        summary = None

    if summary:

        print(f"\n  --- SUMMARY ---")
        print(f"  Mean sag ratio: {summary['mean_sag_ratio']:.3f} +/- {summary['std_sag_ratio']:.3f}")
        print(f"  Range: {summary['min_sag_ratio']:.3f} - {summary['max_sag_ratio']:.3f}")

    return {
        'hyper_sweeps': hyper_sweeps,
        'sag_results': sag_results,
        'summary': summary,
    }



def plot_sag_diagnostics(bundle_path, sweep, times, voltages, stimulus_start, stimulus_end,
                         v_peak, v_steady, v_rest, baseline_start, baseline_end):

    peak_end = stimulus_start + 0.080
    steady_start = stimulus_end - 0.081
    steady_end = stimulus_end - 0.001

    plt.figure(figsize=(8,4))

    plt.plot(times, voltages, color="black", linewidth=1)

    # rest (baseline) window
    plt.axvspan(baseline_start, baseline_end,
                color="purple", alpha=0.2, label="v_rest window")

    # peak window
    plt.axvspan(stimulus_start, peak_end,
                color="green", alpha=0.2, label="v_peak window")

    # steady window
    plt.axvspan(steady_start, steady_end,
                color="orange", alpha=0.2, label="v_steady window")

    # key points
    plt.scatter([],[], label=f"v_rest={v_rest:.2f} mV", color="purple")
    plt.scatter([],[], label=f"v_peak={v_peak:.2f} mV", color="green")
    plt.scatter([],[], label=f"v_steady={v_steady:.2f} mV", color="orange")

    plt.axhline(v_rest, color="purple", linestyle="--")
    plt.axhline(v_peak, color="green", linestyle="--")
    plt.axhline(v_steady, color="orange", linestyle="--")

    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.title("Sag Diagnostic Plot")

    plt.legend()
    plt.tight_layout()
    plot_dir = Path(bundle_path) / "SagCurrent"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir / f"SagCurrent_sweep{sweep}.jpeg", dpi=300)
    plt.close()

# FOR TESTING
# if __name__ == "__main__":
#     # Test on test_bundle2
#     bundle_dir = "test_bundle2/sub-131113"
#     results = calculate_sag_for_bundle(bundle_dir)
    
#     if results:
#         print("\n" + "="*70)
#         print("Results can be integrated into analysis pipeline")
#         print("="*70)
