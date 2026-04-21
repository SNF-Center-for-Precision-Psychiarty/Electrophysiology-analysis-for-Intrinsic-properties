# Sag Current Analysis Guide

## What is Sag Current?

**Sag current** is the voltage response during hyperpolarizing current injection, caused by **HCN (Hyperpolarization-activated Cyclic Nucleotide-gated) channels**. These channels are responsible for regulating the neuron's excitability and input resistance.

## The Physics

When you inject negative (hyperpolarizing) current:

1. **Immediate response (t≈0-5ms):** Voltage rapidly hyperpolarizes (becomes more negative)
2. **Sag phase (t≈5-200ms):** HCN channels open, allowing positive current to flow back in
3. **Recovery phase:** Voltage gradually "sags" or relaxes back toward baseline

The amount and speed of recovery indicates **HCN channel density and kinetics**.

## In Your Pipeline

**Step 6** of the analysis pipeline calculates sag automatically:

```
Step 1: Load data
Step 1.5: Apply 5 kHz low-pass filter
Step 1.6: Visualize filter
Step 2: Sweep configuration
Step 3: RMP (resting membrane potential)
Step 4: Spike detection
Step 5: Savitzky-Golay smoothing
Step 6: Input resistance
[NEW] Step 6: Sag current analysis ← YOU ARE HERE
Step 7: Results finalization
```

## What Gets Measured

### Automatically Identified
- **Negative-current sweeps:** Sweeps with injected current < 0 pA
- **Stimulus window:** From sweep_config.json (typically 10-810 ms)
- **Sag baseline reference (`V_baseline`):** Most negative voltage in the first 80 ms of stimulus

### Calculated Metrics

For each negative-current sweep:

| Metric | Description | Formula / Units |
|--------|-------------|-----------------|
| `sag_voltage_mV` | Absolute sag magnitude (how much voltage recovers from `V_min`) | `V_steady - V_min` (mV) |
| `sag_ratio` | Sag ratio using current baseline definition | `(V_steady - V_min) / (V_baseline - V_min)` (unitless) |
| `sag_percent` | Sag as percentage | `sag_ratio × 100` (%) |

Important:
- With the current definition of `V_baseline` (minimum in first 80 ms of stimulus), `sag_ratio` is not constrained to 0-1 and can exceed 1.
- A value `>1` does not necessarily mean the trace overshot pre-stimulus resting Vm.

### Example from Test Data

For Sweep 0 (-100 pA injection):

```
V_baseline:  -67.62 mV  (minimum within first 80 ms of stimulus)
V_min:       -76.73 mV  (most negative voltage reached)
V_steady:    -67.54 mV  (mean over last 80 ms of stimulus, with 1 ms end buffer)

Total hyperpolarization: 9.11 mV
Sag voltage:             9.19 mV  (recovery from minimum)
Sag ratio:               1.009

Why this can be >1:
- Denominator is `(V_baseline - V_min)`, where `V_baseline` is already close to `V_min` by design.
- If that denominator is small, modest recovery can produce `sag_ratio > 1`.
```

## Interpreting Sag Ratio

```
sag_ratio near 0   →  Little recovery from V_min
sag_ratio around 1 →  Recovery roughly equals (V_baseline - V_min)
sag_ratio > 1      →  Recovery exceeds (V_baseline - V_min) under this definition
```

Interpretation note:
- In this pipeline, treat `sag_ratio` primarily as a within-pipeline comparative metric across sweeps/cells processed the same way.
- Do not assume a hard biological bound at 1 for this specific implementation.

### Your Test Data

Mean sag ratio: **1.047 ± 0.102** (across 5 negative-current sweeps)

**Interpretation:**
- Indicates measurable sag/recovery during hyperpolarizing sweeps.
- Should be interpreted with this pipeline's ratio definition (not as a strict 0-1 normalized fraction).

## In analysis.parquet

Three new columns are automatically added:

```python
# For negative-current sweeps only
df['sag_voltage_mV']   # float, in mV
df['sag_ratio']        # float, unitless
df['sag_percent']      # float, percentage

# For depolarizing sweeps
# All three columns will be NaN (empty)
```

### Accessing Results

```python
import pandas as pd

df = pd.read_parquet("bundle_dir/analysis.parquet")

# Get sag measurements for all sweeps
print(df[['sweep', 'avg_injected_current_pA', 'sag_voltage_mV', 'sag_ratio']])

# Get only negative-current sweeps with sag data
hyper = df[df['avg_injected_current_pA'] < 0]
print(hyper[['sweep', 'sag_ratio', 'sag_percent']])

# Calculate mean sag
mean_sag = hyper['sag_ratio'].mean()
print(f"Mean sag ratio: {mean_sag:.3f}")
```

## Technical Details

### Voltage Measurement Windows

Uses timing from `sweep_config.json`:

```json
{
  "windows": {
    "baseline_start_s": 0.0,
    "baseline_end_s": 0.00999,
    "stimulus_start_s": 0.010020000000000001,
    "stimulus_end_s": 0.8100000000000002
  }
}
```

**Why this matters:**
- Stimulus window = voltage response to current injection
- `V_baseline` for sag ratio = minimum voltage in first 80 ms of stimulus
- `V_steady` = mean voltage in last 80 ms of stimulus, with 1 ms buffer before stimulus end

### Sampling Details

- Sampling rate: 200 kHz (5 µs per sample)
- Early-stimulus baseline reference duration: 80 ms (16,000 samples)
- Stimulus duration: ~800 ms (160,000 samples)
- Steady-state window: Last 80 ms with 1 ms end buffer (~16,000 samples)

## Code Structure

### Main Functions in sag_current.py

```python
find_negative_sweeps(analysis_df, threshold_pA=0)
# Returns: List of sweep numbers with injected current < threshold

measure_voltage_response(mv_data, sweep, sweep_config=None)
# Returns: Dict with v_baseline, v_min, v_steady, t_v_min

calculate_sag(voltage_response)
# Returns: Dict with sag_voltage_mV, sag_ratio, sag_percent

calculate_sag_for_bundle(bundle_dir, verbose=True)
# Returns: Dict with hyper_sweeps, sag_results, summary
# Also prints detailed analysis to console
```

### Integration in run_analysis.py

Called automatically as Step 6 (after input resistance):

```python
sag_results = calculate_sag_for_bundle(bundle_dir, verbose=True)

if sag_results:
    # Add sag columns to analysis.parquet
    df_analysis['sag_voltage_mV'] = np.nan
    df_analysis['sag_ratio'] = np.nan
    df_analysis['sag_percent'] = np.nan
    
    # Fill in values for negative-current sweeps
    for sweep, measurements in sag_results['sag_results'].items():
        mask = df_analysis['sweep'] == sweep
        df_analysis.loc[mask, 'sag_voltage_mV'] = measurements['sag_voltage_mV']
        df_analysis.loc[mask, 'sag_ratio'] = measurements['sag_ratio']
        df_analysis.loc[mask, 'sag_percent'] = measurements['sag_percent']
    
    df_analysis.to_parquet(bundle_path / "analysis.parquet", index=False)
```

## Troubleshooting

### "No negative-current sweeps found"

**Cause:** All injected currents are ≥ 0 pA

**Solution:** Check your sweep_config for actual current values
```python
import json
with open("bundle_dir/sweep_config.json") as f:
    config = json.load(f)
    for sweep_id, sweep_info in config['sweeps'].items():
        print(f"Sweep {sweep_id}: {sweep_info['stimulus_level_pA']} pA")
```

### Sag ratio looks wrong

**Check:**
1. Is sweep_config.json present in the bundle?
2. Are stimulus windows reasonable (typically 10-800+ ms)?
3. Remember: with this implementation, `sag_ratio > 1` can be valid and does not automatically indicate an error.

### Missing sag columns in analysis.parquet

**Solution:** Re-run the pipeline for that bundle. Step 6 will:
1. Calculate sag for negative-current sweeps
2. Add the three columns
3. Save updated analysis.parquet

## Biological Significance

### What HCN Channels Do

- **Role:** Control resting excitability and input resistance
- **Location:** Soma and dendrites
- **Activation:** Opens in response to hyperpolarization
- **Function:** Prevent excessive hyperpolarization by letting positive current back in

### Typical Sag Ratios by Cell Type

Note: The values below are literature-style heuristics and may not map directly to this pipeline's current `sag_ratio` definition.
For strict comparisons, use one consistent calculation method across all datasets.

| Cell Type | Typical Sag Ratio |
|-----------|-----------------|
| Fast-spiking interneurons | 0.1-0.3 |
| Regular-spiking pyramidal cells | 0.8-1.2 |
| Neurons with strong Ih | 1.2-2.0 |

Your test data (1.047) falls in a reasonable range for this dataset and this pipeline definition.

## Related Metrics

**Also measured in this pipeline:**
- **Input resistance:** Directly affected by HCN channel state
- **RMP:** Baseline for comparing hyperpolarization
- **Current threshold:** Relates to excitability regulation

## References

- Sag measurement is described in:
  - Hodgkin & Huxley (1952) - Original H-H model
  - Robinson & Siegelbaum (2003) - Ih kinetics
  - Kole et al. (2006) - HCN distribution in neurons

---

