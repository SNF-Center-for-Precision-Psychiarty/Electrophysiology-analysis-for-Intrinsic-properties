"""
Kink detection module for identifying pre-upstroke peaks in spike upstrokes.

Improved version:
- Anchors to main upstroke (max dV/dt)
- Only considers peaks BEFORE main peak
- Uses stronger prominence threshold
- Filters by kink-to-main ratio
- Applies temporal constraint (kink must be close to upstroke)
"""

import numpy as np
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt
from pathlib import Path


# -----------------------------
# Configuration
# -----------------------------
KINK_DETECTION_PROMINENCE_PERCENT = 0.1   # 10% of max dV/dt
KINK_DETECTION_MIN_DISTANCE_SAMPLES = 5   # Increase separation
KINK_DETECTION_MIN_PROMINENCE_FOR_LOCAL_MAXIMA = 0.10  # 10% of max dV/dt - suppress tiny local wiggles
KINK_MIN_SELECTED_PROMINENCE_RATIO = 0.14 # Stage-1 selected prominence ratio requirement
KINK_RATIO_THRESHOLD = 0.45               # Stage-1 kink/main ratio requirement
KINK_MIN_POST_DIP_RATIO = 0.14            # Stage-1 post-kink dip requirement
KINK_MIN_INTERVAL_MS = 0.6                # Kink must precede max upstroke by at least 0.6 ms
KINK_ENABLE_STAGE2 = False                # Disable relaxed fallback stage; keep strict Stage-1 only
KINK_STAGE2_RATIO_RELAX = 0.03            # Stage-2 relaxed ratio amount (tighter fallback)
KINK_STAGE2_PROMINENCE_RELAX = 0.04       # Stage-2 relaxed selected-prominence amount
KINK_STAGE2_DIP_RELAX = 0.03              # Stage-2 relaxed post-dip amount
KINK_TIEBREAK_MIN_MARGIN_RATIO = 0.05     # If top two candidates are closer than this, use tie-break rules
KINK_ANTI_WIGGLE_MIN_WIDTH_MS = 0.30      # Reject ultra-narrow wiggle-like candidates
KINK_ANTI_WIGGLE_MIN_PROMINENCE_RATIO = 0.10  # Reject weak local wiggles
KINK_ANTI_WIGGLE_MAX_SIGN_CHANGES = 6     # Reject highly oscillatory post-candidate segments


# -----------------------------
# Peak detection in dV/dt
# -----------------------------
def find_peaks_in_dvdt(dvdt_array, prominence_percent=KINK_DETECTION_PROMINENCE_PERCENT):
    if len(dvdt_array) < 3:
        return []

    max_dvdt = np.max(dvdt_array)
    if max_dvdt <= 0:
        return []

    min_prominence = prominence_percent * max_dvdt

    peaks, properties = find_peaks(
        dvdt_array,
        prominence=min_prominence,
        distance=KINK_DETECTION_MIN_DISTANCE_SAMPLES
    )

    return peaks, properties


# -----------------------------
# Kink metric computation
# -----------------------------
def measure_kink_metrics(dvdt_array, times_array, threshold_idx, debug=False):
    """
    Measure kink metrics.

    Kink = secondary dV/dt peak BETWEEN spike threshold and max upstroke.
    """

    result = {
        'num_kinks': 0,
        'kink_interval_ms': np.nan,
        'kink_ratio': np.nan,
        'kink_height_dvdt': np.nan,
        'kink_idx': None,
    }

    if len(dvdt_array) < 3:
        if debug:
            print(f"    [KINK] Array too short: {len(dvdt_array)} samples")
        return result

    # --- Step 1: Identify main upstroke ---
    max_upstroke_idx = np.argmax(dvdt_array)
    max_upstroke_height = dvdt_array[max_upstroke_idx]

    if debug:
        print(f"    [KINK] Step 1: Identify main upstroke")
        print(f"      Max upstroke index: {max_upstroke_idx}")
        print(f"      Max upstroke height: {max_upstroke_height:.3f} mV/ms")
        print(f"      Threshold index: {threshold_idx}")

    if max_upstroke_height <= 0:
        if debug:
            print(f"    [KINK] Max upstroke height invalid: {max_upstroke_height}")
        return result

    # --- Step 2: Find candidate peaks ---
    if debug:
        print(f"    [KINK] Step 2: Find candidate peaks with find_peaks()")
        print(f"      Prominence threshold: {KINK_DETECTION_PROMINENCE_PERCENT*100}% = {KINK_DETECTION_PROMINENCE_PERCENT * max_upstroke_height:.3f} mV/ms")
    
    peaks, properties = find_peaks_in_dvdt(dvdt_array)

    if debug:
        print(f"      find_peaks() returned: {len(peaks)} peaks")
        if len(peaks) > 0:
            for i, p in enumerate(peaks):
                prominence = properties['prominences'][i] if 'prominences' in properties else np.nan
                print(f"        Peak {i}: idx={p}, height={dvdt_array[p]:.3f}, ratio={dvdt_array[p]/max_upstroke_height:.3f}, prominence={prominence:.3f}")

    if len(peaks) == 0:
        if debug:
            print(f"      No peaks found by find_peaks()")

    # --- Step 3: Restrict peaks to threshold → upstroke window ---
    # Also check for peaks that are high but might have been excluded by prominence
    if debug:
        print(f"    [KINK] Step 3: Filter to window ({threshold_idx} < idx < {max_upstroke_idx})")
    
    valid_peaks = [
        p for p in peaks
        if threshold_idx < p < max_upstroke_idx
    ]
    
    if debug:
        print(f"      Peaks passing window filter: {len(valid_peaks)}")
        if len(valid_peaks) > 0:
            for p in valid_peaks:
                print(f"        idx={p}, height={dvdt_array[p]:.3f}, ratio={dvdt_array[p]/max_upstroke_height:.3f}")
    
    # Additionally, find ANY local maximum in the window, not just prominent ones
    # This catches peaks that have high absolute height but low prominence
    if debug:
        print(f"    [KINK] Step 3b: Find ALL local maxima in window")
    
    all_local_maxima = []
    for i in range(threshold_idx + 1, max_upstroke_idx):
        if dvdt_array[i] > dvdt_array[i-1] and dvdt_array[i] >= dvdt_array[i+1]:
            all_local_maxima.append(i)
    
    if debug:
        print(f"      Local maxima found: {len(all_local_maxima)}")
        if len(all_local_maxima) > 0:
            for i in all_local_maxima:
                # Calculate prominence for local maxima
                # Find the highest point to the left that is lower than this peak
                left_min = np.min(dvdt_array[threshold_idx:i]) if i > threshold_idx else dvdt_array[threshold_idx]
                # Find the highest point to the right that is lower than this peak
                right_min = np.min(dvdt_array[i+1:max_upstroke_idx+1]) if i < max_upstroke_idx else dvdt_array[max_upstroke_idx]
                prominence = dvdt_array[i] - max(left_min, right_min)
                print(f"        idx={i}, height={dvdt_array[i]:.3f}, ratio={dvdt_array[i]/max_upstroke_height:.3f}, prominence={prominence:.3f}")
    
    # Filter local maxima by prominence: exclude peaks with near-zero prominence
    min_prominence_threshold = KINK_DETECTION_MIN_PROMINENCE_FOR_LOCAL_MAXIMA * max_upstroke_height
    filtered_local_maxima = []
    for i in all_local_maxima:
        left_min = np.min(dvdt_array[threshold_idx:i]) if i > threshold_idx else dvdt_array[threshold_idx]
        right_min = np.min(dvdt_array[i+1:max_upstroke_idx+1]) if i < max_upstroke_idx else dvdt_array[max_upstroke_idx]
        prominence = dvdt_array[i] - max(left_min, right_min)
        if prominence >= min_prominence_threshold:
            filtered_local_maxima.append(i)
    
    if debug and len(filtered_local_maxima) < len(all_local_maxima):
        print(f"      After prominence filter (>{min_prominence_threshold:.3f}): {len(filtered_local_maxima)} peaks remain")
    
    # Combine: use find_peaks results, but also add any high local maxima
    candidate_peaks = set(valid_peaks) | set(filtered_local_maxima)

    if len(candidate_peaks) == 0:
        if debug:
            print(f"    [KINK] NO CANDIDATE PEAKS FOUND - returning empty result")
        return result

    if debug:
        print(f"    [KINK] Step 4: Combined candidates")
        print(f"      Total unique candidates: {len(candidate_peaks)}")

    # --- Step 4: Build simple per-candidate metrics (rule-based path) ---
    if len(times_array) >= 2:
        dt_ms = np.mean(np.diff(times_array)) * 1000.0
    else:
        dt_ms = 0.02

    candidate_records = []
    for p in candidate_peaks:
        height = dvdt_array[p]
        kink_ratio = height / max_upstroke_height

        left_min = np.min(dvdt_array[threshold_idx:p]) if p > threshold_idx else dvdt_array[threshold_idx]
        right_min = np.min(dvdt_array[p + 1:max_upstroke_idx + 1]) if p < max_upstroke_idx else dvdt_array[max_upstroke_idx]
        selected_prominence = height - max(left_min, right_min)
        selected_prominence_ratio = selected_prominence / max_upstroke_height

        if max_upstroke_idx - p >= 2:
            post_segment = dvdt_array[p + 1:max_upstroke_idx]
            post_min = np.min(post_segment) if len(post_segment) > 0 else height
            post_dip_ratio = (height - post_min) / max_upstroke_height
            centered = post_segment - np.mean(post_segment) if len(post_segment) > 0 else np.array([])
            sign_changes = int(np.sum(np.diff(np.sign(centered)) != 0)) if len(centered) >= 3 else 0
        else:
            post_dip_ratio = 0.0
            sign_changes = 0

        kink_interval_ms = abs((times_array[p] - times_array[max_upstroke_idx]) * 1000.0)

        try:
            widths, _, _, _ = peak_widths(dvdt_array, [p], rel_height=0.5)
            width_ms = widths[0] * dt_ms
        except Exception:
            width_ms = np.nan

        candidate_records.append({
            'idx': p,
            'height': height,
            'kink_ratio': kink_ratio,
            'selected_prominence_ratio': selected_prominence_ratio,
            'post_dip_ratio': post_dip_ratio,
            'interval_ms': kink_interval_ms,
            'width_ms': width_ms,
            'sign_changes': sign_changes,
        })

    if debug:
        print(f"    [KINK] Step 5: Candidate metrics table ({len(candidate_records)} candidates)")
        for c in sorted(candidate_records, key=lambda x: x['height'], reverse=True)[:10]:
            print(
                f"      idx={c['idx']}, h={c['height']:.3f}, ratio={c['kink_ratio']:.3f}, "
                f"prom={c['selected_prominence_ratio']:.3f}, dip={c['post_dip_ratio']:.3f}, "
                f"interval={c['interval_ms']:.3f}ms, width={c['width_ms']:.3f}ms, wiggles={c['sign_changes']}"
            )

    # Anti-wiggle guard for noisy fluctuation-heavy candidates.
    anti_wiggle_survivors = [
        c for c in candidate_records
        if c['selected_prominence_ratio'] >= KINK_ANTI_WIGGLE_MIN_PROMINENCE_RATIO
        and (np.isnan(c['width_ms']) or c['width_ms'] >= KINK_ANTI_WIGGLE_MIN_WIDTH_MS)
        and c['sign_changes'] <= KINK_ANTI_WIGGLE_MAX_SIGN_CHANGES
    ]

    if debug:
        print(f"    [KINK] Step 6: Anti-wiggle survivors: {len(anti_wiggle_survivors)}")

    # Stage 1: strict rule-based gates.
    stage1_survivors = [
        c for c in anti_wiggle_survivors
        if c['kink_ratio'] >= KINK_RATIO_THRESHOLD
        and c['selected_prominence_ratio'] >= KINK_MIN_SELECTED_PROMINENCE_RATIO
        and c['post_dip_ratio'] >= KINK_MIN_POST_DIP_RATIO
        and c['interval_ms'] >= KINK_MIN_INTERVAL_MS
    ]

    selected = None

    if stage1_survivors:
        selected = max(stage1_survivors, key=lambda x: x['height'])
        if debug:
            print(f"    [KINK] Stage 1 passed with {len(stage1_survivors)} survivors")
    elif KINK_ENABLE_STAGE2:
        # Stage 2: relaxed gates + deterministic tie-break for borderline candidates.
        ratio_relaxed = max(0.0, KINK_RATIO_THRESHOLD - KINK_STAGE2_RATIO_RELAX)
        prominence_relaxed = max(0.0, KINK_MIN_SELECTED_PROMINENCE_RATIO - KINK_STAGE2_PROMINENCE_RELAX)
        dip_relaxed = max(0.0, KINK_MIN_POST_DIP_RATIO - KINK_STAGE2_DIP_RELAX)

        stage2_survivors = [
            c for c in anti_wiggle_survivors
            if c['kink_ratio'] >= ratio_relaxed
            and c['selected_prominence_ratio'] >= prominence_relaxed
            and c['post_dip_ratio'] >= dip_relaxed
            and c['interval_ms'] >= KINK_MIN_INTERVAL_MS
        ]

        if debug:
            print(f"    [KINK] Stage 2 survivors (relaxed): {len(stage2_survivors)}")

        if stage2_survivors:
            stage2_sorted = sorted(stage2_survivors, key=lambda x: x['height'], reverse=True)
            selected = stage2_sorted[0]
            if len(stage2_sorted) > 1:
                top = stage2_sorted[0]
                second = stage2_sorted[1]
                margin = top['kink_ratio'] - second['kink_ratio']
                if margin < KINK_TIEBREAK_MIN_MARGIN_RATIO:
                    # Tie-break: prefer deeper dip, then stronger prominence, then fewer wiggles.
                    selected = sorted(
                        stage2_sorted[:2],
                        key=lambda x: (x['post_dip_ratio'], x['selected_prominence_ratio'], -x['sign_changes']),
                        reverse=True,
                    )[0]
                    if debug:
                        print("    [KINK] Tie-break used for near-equal candidates")
    elif debug:
        print("    [KINK] Stage 2 disabled; strict Stage 1 only")

    if selected is None:
        if debug:
            print("    [KINK] No candidate passed stage rules")
        return result

    kink_idx = selected['idx']
    kink_height = selected['height']
    kink_ratio = selected['kink_ratio']
    kink_interval_ms = selected['interval_ms']

    if debug:
        print(f"    [KINK] Selected idx={kink_idx}, ratio={kink_ratio:.3f}, interval={kink_interval_ms:.3f} ms")
        print("    [KINK] KINK DETECTED")

    # --- Step 7: Update results ---
    result.update({
        'num_kinks': 1,
        'kink_interval_ms': kink_interval_ms,
        'kink_ratio': kink_ratio,
        'kink_height_dvdt': kink_height,
        'kink_idx': kink_idx
    })

    return result

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
    
    # Need at least 3 spikes to establish a trend
    if len(spike_amplitudes) < 3:
        return False
    
    # Calculate median amplitude of all spikes in sweep (robust to outliers)
    median_amplitude = np.median(spike_amplitudes)
    
    # Get current spike amplitude
    current_amplitude = spike_amplitudes[current_spike_index]
    
    # Check if this spike is in the last 10% of sweep and is significantly smaller
    last_spike_fraction = 0.1
    last_spike_index = int(len(spike_amplitudes) * (1 - last_spike_fraction))
    
    # Only apply struggling detection to last 10% of spikes (more conservative)
    if current_spike_index >= last_spike_index:
        amplitude_ratio = current_amplitude / median_amplitude
        
        # Skip if amplitude is <threshold percent of median (cell struggling)
        if amplitude_ratio < (amplitude_threshold_percent / 100):
            return True
    
    return False


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
            print(f"    [WRAPPER] ✗ Invalid arrays: len(v)={len(voltages)}, len(t)={len(times)}, len(dvdt)={len(dvdt)}")
        return {
            'num_kinks': 0,
            'kink_interval_ms': np.nan,
            'kink_ratio': np.nan,
            'kink_height_dvdt': np.nan,
            'kink_idx': None
        }

    # threshold is at index 0, upstroke is at last index
    threshold_local = 0
    upstroke_local = len(dvdt) - 1 #this should be 63

    if debug:
        print(f"    Calling measure_kink_metrics with:")
        print(f"      threshold_local={threshold_local}")
        print(f"      upstroke_local={upstroke_local}")
        print(f"      dV/dt at upstroke_local index {upstroke_local}: {dvdt[upstroke_local]:.3f} mV/ms")
        print(f"      Max dV/dt in window (aka upstroke): {np.max(dvdt):.3f} mV/ms at index {np.argmax(dvdt)}")

    # run kink detection
    kink_metrics = measure_kink_metrics(
        dvdt,
        times,
        threshold_local,
        debug=debug
    )

    return kink_metrics


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

    # CRITICAL: Only plot from threshold to upstroke (+ small margin)
    # This ensures we're looking at the same spike, not bleeding into the next one
    w_start = threshold_idx
    w_end = upstroke_idx + int(0.5 * (upstroke_idx - threshold_idx))  # Extend 50% past upstroke
    w_end = min(w_end, len(voltages) - 1)  # Ensure we don't exceed array bounds
    
    time_window = times[w_start:w_end+1]
    voltage_window = voltages[w_start:w_end+1]
    
    # Calculate relative times (0 at threshold)
    time_rel = (time_window - time_window[0]) * 1000  # Convert to ms
    
    # Calculate local indices within the window
    threshold_local = 0  # Always at start
    kink_local = kink_idx - w_start
    upstroke_local = upstroke_idx - w_start
    peak_local = peak_idx - w_start
    
    # Validate indices are within window bounds
    if kink_local < 0 or kink_local >= len(time_rel):
        kink_local = max(0, min(kink_local, len(time_rel) - 1))
    if upstroke_local < 0 or upstroke_local >= len(time_rel):
        upstroke_local = max(0, min(upstroke_local, len(time_rel) - 1))
    if peak_local < 0 or peak_local >= len(time_rel):
        peak_local = None  # Peak might be outside this window
    
    # Calculate dV/dt
    dvdt = np.gradient(voltage_window, time_window) * 1000
    
    # Create figure with better layout
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # --- Top: Voltage plot ---
    axes[0].plot(time_rel, voltage_window, 'k-', linewidth=1.5, label='Voltage')
    
    # Mark key points
    axes[0].scatter(time_rel[threshold_local], voltage_window[threshold_local],
                    s=100, color='green', zorder=5, label='Threshold', marker='o')
    axes[0].scatter(time_rel[kink_local], voltage_window[kink_local],
                    s=100, color='orange', zorder=5, label='Kink', marker='s')
    axes[0].scatter(time_rel[upstroke_local], voltage_window[upstroke_local],
                    s=100, color='red', zorder=5, label='Max upstroke', marker='^')
    if peak_local is not None and peak_local >= 0 and peak_local < len(time_rel):
        axes[0].scatter(time_rel[peak_local], voltage_window[peak_local],
                        s=100, color='blue', zorder=5, label='Peak', marker='D')
    
    # Shade the threshold-to-upstroke region (where kink should be)
    axes[0].axvspan(time_rel[threshold_local], time_rel[upstroke_local], 
                   alpha=0.1, color='yellow', label='Kink search window')
    
    axes[0].set_ylabel('Voltage (mV)', fontsize=11, fontweight='bold')
    axes[0].set_title(f'Kink Detection: {spike_id} (Threshold→Upstroke)', fontsize=12, fontweight='bold')
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # --- Bottom: dV/dt plot ---
    axes[1].plot(time_rel, dvdt, color='purple', linewidth=1.5, label='dV/dt')
    
    axes[1].scatter(time_rel[kink_local], dvdt[kink_local],
                    s=100, color='orange', zorder=5, marker='s')
    axes[1].scatter(time_rel[upstroke_local], dvdt[upstroke_local],
                    s=100, color='red', zorder=5, marker='^')
    
    # Draw the width measurement line at 50% of kink peak height
    from scipy.signal import peak_widths
    try:
        if kink_local > 0 and kink_local < len(dvdt) - 1:
            widths, width_height, left_idx, right_idx = peak_widths(dvdt, [kink_local], rel_height=0.5)
            left_idx_int = int(left_idx[0])
            right_idx_int = int(right_idx[0])
            if 0 <= left_idx_int < len(time_rel) and 0 <= right_idx_int < len(time_rel):
                # Draw horizontal line at 50% height
                axes[1].hlines(width_height[0], time_rel[left_idx_int], time_rel[right_idx_int],
                              colors='cyan', linewidth=2.5, linestyle='--', label=f'Width @ 50% = {widths[0]*(time_window[1]-time_window[0])*1000:.3f} ms')
                # Mark the boundaries
                axes[1].scatter([time_rel[left_idx_int], time_rel[right_idx_int]], 
                               [width_height[0], width_height[0]],
                               s=80, color='cyan', zorder=6, marker='|')
    except Exception as e:
        pass
    
    # Shade the threshold-to-upstroke region
    axes[1].axvspan(time_rel[threshold_local], time_rel[upstroke_local], 
                   alpha=0.1, color='yellow')
    
    axes[1].set_ylabel('dV/dt (mV/ms)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Time relative to threshold (ms)', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best', fontsize=9)

    plt.tight_layout()

    plt.savefig(output_dir / f"kink_spike_{spike_id}.jpeg", dpi=200, bbox_inches='tight')
    plt.close()
