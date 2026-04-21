# Electrophysiology Analysis Pipeline (NWB + ABF)

A unified pipeline for analyzing **intracellular electrophysiology recordings** from both:

- **NWB (Neurodata Without Borders)** files  
- **ABF (Axon Binary Format)** files  

The pipeline performs **sweep classification, spike detection, membrane property analysis, and visualization** for current-clamp and voltage-clamp experiments.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Supported File Types](#supported-file-types)
- [Quick Start](#quick-start)
- [Input Requirements](#input-requirements)
- [Pipeline Workflow](#pipeline-workflow)
- [Output Structure](#output-structure)
- [Configuration](#configuration)
- [Memory Optimization](#memory-optimization)

---

## Overview

This pipeline processes electrophysiology recordings to extract:

- **Spike features** (threshold, peak, width, dV/dt)
- **Resting membrane potential (RMP)**
- **Input resistance**
- **Sag current (HCN activity)**
- **Sweep quality classification**
- **Kink detection (pre-upstroke features)**

It is designed to handle **real-world noisy biological data**, including:
- multi-sweep recordings  
- variable sampling rates  
- mixed protocol experiments  

---

## Supported File Types

### NWB (Neurodata Without Borders)
- Standardized neuroscience format
- Used in Allen Brain Observatory, Zuckerman Lab, etc.
- Supports:
  - single protocol (voltage-only / current-only)
  - mixed protocol (stimulus + response)

### ABF (Axon Binary Format)
- Common in patch-clamp experiments
- Typically noisier and less standardized than NWB
- Requires more flexible filtering and classification


---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data
Place NWB files in the workspace directory:
```
Sneha_NWB_Allen_brain/
├── sub-1000610030_ses-1002181694_icephys.nwb  # Single protocol files
├── sub-1000610030_ses-1002208108_icephys.nwb
└── sub-1000610030/                             # Or organize in folders
    └── *.nwb
```
Place ABF files in the workspace directory:
```
data/
├── cell_001.abf
├── cell_002.abf
└── experiment_day_1/
    └── *.abf
```

Or place a mix of both file types in the same workspace directory:
```
data/
├── nwb_files/
│   └── *.nwb
└── abf_files/
    └── *.abf
```

### 3. Run the Pipeline
```bash
python main.py
```

The script will:
- Auto-detect protocol type (single vs mixed)
- Process all `.nwb` files
- Create output folders with analysis results

The pipeline will:
- Detect file type (NWB vs ABF)
  
For NWB:
- Auto-detect protocol (single vs mixed)
- Extract and structure data into bundles
  
For ABF:
- Parse sweeps and standardize format

Run full analysis:
- Sweep classification
- RMP calculation
- Spike detection
- Kink detection
- Input resistance
- Sag analysis

Generate outputs:
- .parquet / .csv results
- plots and summary visualizations
- per-bundle analysis folders

---

## Input Requirements

### NWB File Structure

**Single Protocol** files must contain:
- `acquisition/` - Voltage or current timeseries data
- `stimulus/` - Stimulus waveforms
- `icephys_sequential_recordings` - Metadata about sweeps

**Mixed Protocol** files must contain:
- `acquisition/` - Voltage response timeseries
- `stimulus/` - Current stimulus timeseries
- Both stimulus and response indexed by sweep number

### Sampling Rate
- Typically 200 kHz for stimulus, 50 kHz for response (mixed protocol)
- Can vary; pipeline auto-detects from NWB metadata

### ABF File Structure

ABF files are typically less standardized but must contain:
- Multiple sweeps of voltage recordings
- Associated current injection traces (for current-clamp experiments)
  
Expected Properties
- Consistent sampling rate across sweeps
- Clearly defined stimulus periods (current injection)
- Sufficient baseline (pre-stimulus) region
  
Notes:
- ABF data is often noisier than NWB
- The pipeline applies more flexible filtering and validation
- No strict schema is required (unlike NWB)

### General Requirements (All File Types)
- Recordings should be current-clamp (not voltage-clamp)
- Each sweep must contain:
  - Baseline (no current injection)
  - Stimulus (current injection)
- Data must be long enough to:
  - Detect spikes
  - Compute baseline statistics (e.g., RMP)

---

## Pipeline Workflow

### Step 1: Sweep Classification (`sweep_classifier.py`)

**Purpose**: Identify valid current-clamp sweeps and filter out artifacts/malformed data

**What it does**:
1. Detects stimulus windows by finding stable current injection periods
2. Identifies baseline (no stimulus) and response windows
3. Filters out:
   - Voltage-clamp sweeps (not current-clamp)
   - Artifact sweeps (sharp corners, voltage jumps)
   - Malformed sweeps (insufficient baseline/stimulus)

**Key outputs**:
- `sweep_config.json` - Contains stimulus/baseline/response windows for each sweep
- Classification plots showing kept vs dropped sweeps

**Configuration** (see `analysis_config.py`)

---

### Step 2: Data Extraction

**Single Protocol** (`process_human_data.py`):
- Extracts voltage and current data from `acquisition/` and `stimulus/`
- Separates by unit type (mV vs pA)
- Creates unified tables: `voltage_all.parquet`, `current_all.parquet`

**Mixed Protocol** (`process_human_data_mixed_protocol.py`):
- Extracts stimulus (current) and response (voltage) separately
- Filters by unit type to handle mixed data
- Creates four tables:
  - `stimulus_all.parquet` - All stimulus data
  - `response_all.parquet` - All response data
  - `stimulus_currentclamp.parquet` - Current (pA) only
  - `response_voltageclamp.parquet` - Voltage (mV) only
 
**ABF Files** (`zuckerman-abf.py`):
- Loads sweeps directly from .abf files
- Extracts:
  - Voltage traces (mV)
  - Current injection traces (pA)
- Standardizes structure to match NWB pipeline format
- Outputs:
  - `voltage_all.parquet`
  - `current_all.parquet`
  
This standardization allows downstream analysis (spike detection, RMP, etc.) to run identically for both ABF and NWB data.

**Memory optimization**: Uses save→delete→read-back pattern to avoid holding full dataset in RAM

---

### Step 3: Action Potential Detection (`spike_detection_new.py`)

**Purpose**: Detect and characterize action potential morphology

**What it does**:
1. Smooths voltage traces using adaptive Savitzky-Golay filter
2. Calculates dV/dt (voltage derivative)
3. Finds peaks using `scipy.signal.find_peaks`
4. For each spike, extracts:
   - **Peak** - Maximum voltage
   - **Threshold** - Where dV/dt exceeds 5% of max
   - **Fast trough** - Minimum voltage after peak
   - **Upstroke/downstroke velocity** - Max/min dV/dt
   - **Width** - Duration at half-max amplitude
   - **Latency** - Time from stimulus onset to first spike

**Configuration** (see `analysis_config.py`)

**Outputs**:
- `AP_analysis.csv` - Per-spike features (peak, threshold, width, etc.)
- `AP_Per_Sweep/` - Individual spike plots per sweep
- `Averaged_Peaks_Per_Sweep/` - Averaged AP waveforms

---

### Step 4: Resting Membrane Potential (`sav_gol_filter.py`)

**Purpose**: Calculate RMP and baseline stability during non-stimulus periods

**What it does**:
1. Filters to baseline periods only (no stimulus)
2. Applies Savitzky-Golay smoothing to remove noise
3. Downsamples to 25ms windows
4. Calculates:
   - **Filtered RMP** - Mean baseline voltage
   - **RMP derivative** - Rate of voltage drift (mV/s)
   - **Voltage range** - Max - min during baseline
   - **Std dev** - Baseline noise/stability

**Why this matters**:
- Detects unhealthy cells (drifting baseline)
- Identifies seal quality issues
- Flags sweeps with excessive noise

**Configuration** (see `analysis_config.py`)

**Outputs**:
- RMP metrics appended to `analysis.csv`
- `Sav_Gol_Plots_Per_Sweep/` - Visualization of filtered baseline

---

### Step 5: Input Resistance (`input_resistance.py`)

**Purpose**: Calculate membrane input resistance from hyperpolarizing current steps

**What it does**:
1. Identifies negative-current sweeps (below 0 pA)
2. Finds minimum voltage during stimulus (max hyperpolarization)
3. Calculates ΔV (voltage deflection from baseline)
4. Calculates R_input = ΔV / I_injected (Ohm's law)

**Why this matters**:
- Measures membrane integrity
- Indicates cell health
- Key parameter for computational models

**Outputs**:
- Input resistance values in `analysis.csv`
- `Input_Resistance/` - Voltage response plots for hyperpolarizing steps

---

### Step 6: Summary Outputs

**For each NWB/ABF file, creates**:
- `analysis.csv` - Per-sweep summary (RMP, input resistance, spike count, etc.)
- `AP_analysis.csv` - Per-spike features (all detected action potentials)
- `manifest.json` - Metadata about the analysis run
- `sweep_config.json` - Stimulus/baseline/response windows for each sweep
- `all_plots_summary.pdf` - Combined visualization of all plots
- `min_frequency.csv` - Row with minimum current that produces spikes
- `max_frequency.csv` - Row with maximum spike frequency
- `mean_frequency.csv` - Mean of all rows up to max frequency

**Grid plots**:
- `voltage_grid.jpeg` / `current_grid.jpeg` (single protocol)
- `stimulus_grid.jpeg` / `response_grid.jpeg` (mixed protocol)
- Show all valid sweeps in grid layout

---

## Output Structure
```
bundle_dir/
├── analysis.parquet
├── AP_analysis.csv
├── sweep_config.json
├── manifest.json
├── plots/
│   ├── AP_Per_Sweep/
│   ├── Averaged_Peaks_Per_Sweep/
│   ├── Sav_Gol_Plots_Per_Sweep/
│   ├── Input_Resistance/
│   └── grid_plots/
```

---

## Configuration

### Central Configuration File: `analysis_config.py`

All analysis parameters are centralized for easy tuning

**To modify thresholds**: Edit `analysis_config.py` - changes apply globally

---

## Memory Optimization

The pipeline includes aggressive memory management for processing large datasets:

### Key Optimizations (see `MEMORY_OPTIMIZATION.md`):

1. **Garbage collection** - `gc.collect()` after each file
2. **Explicit cleanup** - `del dataframe` after saving to disk
3. **Read-from-disk strategy** - For unified tables:
   - Save stimulus → delete → gc
   - Save response → delete → gc
   - Read back only needed columns for filtering
   - Reduces peak memory by ~70%

4. **Parquet format** - Compressed columnar storage for efficient I/O

### Typical Memory Usage:
- **Single file (small)**: 500 MB - 1 GB
- **Single file (large)**: 2-4 GB
- **Mixed protocol**: 3-6 GB (larger due to dual data streams)

**If you encounter memory issues**:
- Close other applications
- Process files one at a time
- See `MEMORY_OPTIMIZATION.md` for detailed troubleshooting

---

## Troubleshooting

### Common Issues

**1. "System has run out of application memory"**
- See `MEMORY_OPTIMIZATION.md`
- Try processing fewer files at once
- Restart Python kernel between runs

**2. "No valid sweeps found"**
- Check NWB file has current-clamp data (not voltage-clamp)
- Verify stimulus duration ≥ 300ms
- Review `sweep_config.json` to see why sweeps were dropped

**3. "Baseline duration < 25ms"**
- Sweeps need ≥50ms baseline for RMP derivative calculation
- Short baseline sweeps will have NaN for RMP derivative
- This is normal for some protocols - sweeps still analyzed for spikes

**4. Poor PDF quality**
- Individual plots saved as JPG at 100 DPI
- For higher quality, increase DPI in `sweep_classifier.py`
- Trade-off: larger file sizes

---

## File Descriptions

### Core Pipeline Files
- `main.py` - Entry point; orchestrates the entire pipeline
- `process_human_data.py` - Processes single protocol NWB files
- `process_human_data_mixed_protocol.py` - Processes mixed protocol NWB files
- `sweep_classifier.py` - Sweep classification and window detection
- `spike_detection_new.py` - Action potential detection and analysis
- `sav_gol_filter.py` - RMP calculation using Savitzky-Golay filtering
- `input_resistance.py` - Input resistance calculation
- `analysis.py` - Helper functions for data extraction

### Configuration
- `analysis_config.py` - Central parameter configuration
- `requirements.txt` - Python package dependencies

---

## Citation & Data Sources

### Allen Brain Observatory
- Human intracellular electrophysiology data
- Single protocol recordings (voltage OR current)
- https://portal.brain-map.org/

### Zuckerman Institute
- Mixed protocol recordings (voltage AND current)
- Custom experimental protocols