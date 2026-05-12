"""
Kink detection module for identifying pre-upstroke features in spike upstrokes.

Both kink types (local maximum and inflection-point shoulder) produce a valley
in d(dV/dt).  A single algorithm finds that valley and applies three gates:
ratio, depth, and interval.
"""

import numpy as np
from scipy.signal import peak_widths
import matplotlib.pyplot as plt
from pathlib import Path


# -----------------------------
# Configuration
# -----------------------------
KINK_RATIO_THRESHOLD     = 0.25   # kink dV/dt ≥ 25% of max upstroke
KINK_DEPTH_RATIO_THRESHOLD = 0.40 # valley in d(dV/dt) ≥ 40% below surrounding peak acceleration
KINK_MIN_INTERVAL_MS     = 0.04   # kink must precede max upstroke by ≥ 0.04 ms (~2 samples @ 50kHz)


# -----------------------------
# Kink metric computation
# -----------------------------
def measure_kink_metrics(dvdt_array, times_array, threshold_idx, debug=False):
    """
    Detect a kink between spike threshold and max upstroke.

    Works for both kink types:
    - Local-max kink: dV/dt rises, peaks, dips → d2 valley is negative (d2 crosses zero)
    - Inflection kink: dV/dt rises monotonically but has a shoulder → d2 valley is positive minimum

    In both cases the deepest interior local minimum of d2 = np.diff(dV/dt) identifies
    the kink position.  Three gates are applied: ratio, depth_ratio, interval.
    """
    result = {
        'num_kinks': 0,
        'kink_interval_ms': np.nan,
        'kink_ratio': np.nan,
        'kink_height_dvdt': np.nan,
        'kink_idx': None,
    }

    if len(dvdt_array) < 3:
        return result

    max_upstroke_idx    = np.argmax(dvdt_array)
    max_upstroke_height = dvdt_array[max_upstroke_idx]

    if debug:
        print(f"    [KINK] max_upstroke_idx={max_upstroke_idx}, height={max_upstroke_height:.3f} mV/ms")

    if max_upstroke_height <= 0:
        return result

    # Search window: threshold (inclusive) → max upstroke (exclusive)
    search_dvdt = dvdt_array[threshold_idx:max_upstroke_idx]
    n = len(search_dvdt)

    if n < 3:
        if debug:
            print(f"    [KINK] Window too narrow ({n} samples between threshold and upstroke)")
        return result

    # Second differences — proportional to d²V/dt²
    d2 = np.diff(search_dvdt)

    if len(d2) < 3:
        if debug:
            print(f"    [KINK] d2 too short ({len(d2)} samples)")
        return result

    # Deepest interior local minimum in d2 (excluding first and last endpoints)
    valley_idx = None
    for i in range(1, len(d2) - 1):
        if d2[i] < d2[i - 1] and d2[i] < d2[i + 1]:
            if valley_idx is None or d2[i] < d2[valley_idx]:
                valley_idx = i

    if valley_idx is None:
        if debug:
            print("    [KINK] No interior valley in d(dV/dt)")
        return result

    # Depth of valley relative to surrounding peak acceleration
    d2_left_max  = np.max(d2[:valley_idx])     if valley_idx > 0          else d2[0]
    d2_right_max = np.max(d2[valley_idx + 1:]) if valley_idx < len(d2) - 1 else d2[-1]
    d2_normal    = max(d2_left_max, d2_right_max)

    if d2_normal <= 0:
        if debug:
            print(f"    [KINK] d2_normal ≤ 0, no valid acceleration context")
        return result

    depth_ratio = (d2_normal - d2[valley_idx]) / d2_normal

    # Map valley back to dvdt_array coordinates.
    # For local-max kinks: snap backwards to the dV/dt local maximum that precedes
    # the d2 valley — that is the actual kink peak, not the dip after it.
    # For inflection kinks (monotonic rise with shoulder): no local max precedes the
    # valley, so the loop finds nothing and valley_idx is kept as-is.
    n_search = len(search_dvdt)
    kink_pos_local = valley_idx
    for j in range(min(valley_idx, n_search - 1), 0, -1):
        if search_dvdt[j] > search_dvdt[j - 1] and search_dvdt[j] >= search_dvdt[j + 1]:
            kink_pos_local = j
            break

    kink_idx      = threshold_idx + kink_pos_local
    kink_dvdt_val = dvdt_array[kink_idx]
    kink_ratio    = kink_dvdt_val / max_upstroke_height
    kink_interval_ms = abs((times_array[kink_idx] - times_array[max_upstroke_idx]) * 1000.0)

    if debug:
        print(f"    [KINK] valley_idx={valley_idx}, kink_pos_local={kink_pos_local}, kink_idx={kink_idx}")
        print(f"    [KINK] kink_ratio={kink_ratio:.3f}, depth_ratio={depth_ratio:.3f}, interval={kink_interval_ms:.3f}ms")

    # Gates
    if kink_ratio < KINK_RATIO_THRESHOLD:
        if debug:
            print(f"    [KINK] FAIL ratio: {kink_ratio:.3f} < {KINK_RATIO_THRESHOLD}")
        return result
    if depth_ratio < KINK_DEPTH_RATIO_THRESHOLD:
        if debug:
            print(f"    [KINK] FAIL depth: {depth_ratio:.3f} < {KINK_DEPTH_RATIO_THRESHOLD}")
        return result
    if kink_interval_ms < KINK_MIN_INTERVAL_MS:
        if debug:
            print(f"    [KINK] FAIL interval: {kink_interval_ms:.3f}ms < {KINK_MIN_INTERVAL_MS}ms")
        return result

    if debug:
        print("    [KINK] KINK DETECTED")

    result.update({
        'num_kinks': 1,
        'kink_interval_ms': kink_interval_ms,
        'kink_ratio': kink_ratio,
        'kink_height_dvdt': kink_dvdt_val,
        'kink_idx': kink_idx,
    })

    return result


# -----------------------------
# Struggling-cell guard
# -----------------------------
def should_skip_kink_detection_struggling_cell(spike_amplitudes, current_spike_index, amplitude_threshold_percent=60):
    """
    Determine if this spike should be skipped from kink detection because the cell is struggling.

    Args:
        spike_amplitudes: List of spike amplitudes in the sweep (V_peak - V_threshold)
        current_spike_index: Index of current spike in the list
        amplitude_threshold_percent: Skip if spike is <60% of median amplitude

    Returns:
        True if spike should be skipped (cell struggling), False otherwise
    """
    if len(spike_amplitudes) < 3:
        return False

    median_amplitude = np.median(spike_amplitudes)
    current_amplitude = spike_amplitudes[current_spike_index]

    last_spike_fraction = 0.1
    last_spike_index = int(len(spike_amplitudes) * (1 - last_spike_fraction))

    if current_spike_index >= last_spike_index:
        amplitude_ratio = current_amplitude / median_amplitude
        if amplitude_ratio < (amplitude_threshold_percent / 100):
            return True

    return False


# -----------------------------
# Wrapper for full spike
# -----------------------------
def measure_kink_for_spike(voltages, times, dvdt, debug=False):
    """
    Measure kink metrics for a single spike.

    Args:
        voltages: Voltage array from threshold to upstroke (already extracted window)
        times: Time array from threshold to upstroke (same window)
        dvdt: dV/dt array from threshold to upstroke (same window)
        debug: Print debug info
    """
    if debug:
        print(f"  [WRAPPER] measure_kink_for_spike()")
        print(f"    Window size: {len(voltages)} samples")
        print(f"    dV/dt range: {np.min(dvdt):.3f} to {np.max(dvdt):.3f} mV/ms")

    if len(voltages) < 2 or len(times) < 2 or len(dvdt) < 2:
        if debug:
            print(f"    [WRAPPER] Invalid arrays: len(v)={len(voltages)}, len(t)={len(times)}, len(dvdt)={len(dvdt)}")
        return {
            'num_kinks': 0,
            'kink_interval_ms': np.nan,
            'kink_ratio': np.nan,
            'kink_height_dvdt': np.nan,
            'kink_idx': None
        }

    # threshold is at index 0; measure_kink_metrics finds max upstroke internally
    return measure_kink_metrics(dvdt, times, threshold_idx=0, debug=debug)


def plot_kink_diagnostics(
    voltages,
    times,
    threshold_idx,
    kink_idx,
    upstroke_idx,
    peak_idx,
    output_dir,
    spike_id
):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Only plot from threshold to upstroke (+ small margin)
    w_start = threshold_idx
    w_end = upstroke_idx + int(0.5 * (upstroke_idx - threshold_idx))
    w_end = min(w_end, len(voltages) - 1)

    time_window   = times[w_start:w_end + 1]
    voltage_window = voltages[w_start:w_end + 1]

    time_rel = (time_window - time_window[0]) * 1000  # ms, relative to threshold

    threshold_local = 0
    kink_local      = kink_idx - w_start
    upstroke_local  = upstroke_idx - w_start
    peak_local      = peak_idx - w_start

    if kink_local < 0 or kink_local >= len(time_rel):
        kink_local = max(0, min(kink_local, len(time_rel) - 1))
    if upstroke_local < 0 or upstroke_local >= len(time_rel):
        upstroke_local = max(0, min(upstroke_local, len(time_rel) - 1))
    if peak_local < 0 or peak_local >= len(time_rel):
        peak_local = None

    dvdt = np.gradient(voltage_window, time_window) * 1000

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # --- Top: Voltage ---
    axes[0].plot(time_rel, voltage_window, 'k-', linewidth=1.5, label='Voltage')
    axes[0].scatter(time_rel[threshold_local], voltage_window[threshold_local],
                    s=100, color='green', zorder=5, label='Threshold', marker='o')
    axes[0].scatter(time_rel[kink_local], voltage_window[kink_local],
                    s=100, color='orange', zorder=5, label='Kink', marker='s')
    axes[0].scatter(time_rel[upstroke_local], voltage_window[upstroke_local],
                    s=100, color='red', zorder=5, label='Max upstroke', marker='^')
    if peak_local is not None and 0 <= peak_local < len(time_rel):
        axes[0].scatter(time_rel[peak_local], voltage_window[peak_local],
                        s=100, color='blue', zorder=5, label='Peak', marker='D')
    axes[0].axvspan(time_rel[threshold_local], time_rel[upstroke_local],
                    alpha=0.1, color='yellow', label='Kink search window')
    axes[0].set_ylabel('Voltage (mV)', fontsize=11, fontweight='bold')
    axes[0].set_title(f'Kink Detection: {spike_id} (Threshold→Upstroke)', fontsize=12, fontweight='bold')
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # --- Bottom: dV/dt ---
    axes[1].plot(time_rel, dvdt, color='purple', linewidth=1.5, label='dV/dt')
    axes[1].scatter(time_rel[kink_local], dvdt[kink_local],
                    s=100, color='orange', zorder=5, marker='s')
    axes[1].scatter(time_rel[upstroke_local], dvdt[upstroke_local],
                    s=100, color='red', zorder=5, marker='^')

    try:
        from scipy.signal import peak_widths as _pw
        if 0 < kink_local < len(dvdt) - 1:
            widths, width_height, left_idx, right_idx = _pw(dvdt, [kink_local], rel_height=0.5)
            li, ri = int(left_idx[0]), int(right_idx[0])
            if 0 <= li < len(time_rel) and 0 <= ri < len(time_rel):
                axes[1].hlines(width_height[0], time_rel[li], time_rel[ri],
                               colors='cyan', linewidth=2.5, linestyle='--',
                               label=f'Width @ 50% = {widths[0] * (time_window[1] - time_window[0]) * 1000:.3f} ms')
                axes[1].scatter([time_rel[li], time_rel[ri]], [width_height[0], width_height[0]],
                                s=80, color='cyan', zorder=6, marker='|')
    except Exception:
        pass

    axes[1].axvspan(time_rel[threshold_local], time_rel[upstroke_local],
                    alpha=0.1, color='yellow')
    axes[1].set_ylabel('dV/dt (mV/ms)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Time relative to threshold (ms)', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / f"kink_spike_{spike_id}.jpeg", dpi=200, bbox_inches='tight')
    plt.close()
