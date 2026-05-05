#!/usr/bin/env python3
"""
ELECTROPHYSIOLOGY ANALYSIS PIPELINE
====================================

Main entry point for analyzing electrophysiology data.
Supports both ABF (patch clamp) and NWB (Neurodata Without Borders) formats.

This script will guide you through the analysis process step-by-step.
"""
11
import sys
import os
import subprocess
from pathlib import Path
import h5py
import json
import gc
from typing import Tuple, Dict, Any


def detect_nwb_protocol_type(nwb_file: str) -> Tuple[str, Dict[str, Any]]:
    """
    Detect if NWB file has single or mixed protocols.
    
    Returns:
        Tuple of (protocol_type, protocol_info) where:
        - protocol_type: "single" or "mixed"
        - protocol_info: dict with protocol details
    """
    try:
        with h5py.File(nwb_file, 'r') as f:
            # Check if this is an intracellular ephys recording
            if 'general' not in f or 'intracellular_ephys' not in f['general']:
                return None, None
            
            ice = f['general']['intracellular_ephys']
            protocols_found = set()
            rates_found = set()
            
            # Check sweep table for protocol information
            if 'sweep_table' in ice:
                sweep_table = ice['sweep_table']
                
                # Look at the intracellular_electrode column
                if 'intracellular_electrode' in sweep_table:
                    electrodes = sweep_table['intracellular_electrode']
                    
                    for i, electrode_ref in enumerate(electrodes[:]):
                        # Get protocol from stimulus description or other attributes
                        if 'stimulus_description' in electrodes.attrs:
                            stim = electrodes.attrs['stimulus_description']
                            protocols_found.add(stim.decode() if isinstance(stim, bytes) else stim)
            
            # Check stimulus data for protocol types (VoltageClamp vs CurrentClamp)
            stimulus_data = f['stimulus/presentation'] if 'stimulus' in f and 'presentation' in f['stimulus'] else None
            
            if stimulus_data:
                for key in stimulus_data.keys():
                    series = stimulus_data[key]
                    series_type = series.attrs.get('neurodata_type', b'').decode() if isinstance(series.attrs.get('neurodata_type'), bytes) else series.attrs.get('neurodata_type', '')
                    if 'VoltageClampStimulusSeries' in series_type or 'VoltageClamp' in key:
                        protocols_found.add('VoltageClamp')
                    elif 'CurrentClampStimulusSeries' in series_type or 'CurrentClamp' in key:
                        protocols_found.add('CurrentClamp')
            
            # Check acquisition data for response types (which tells us about stimulus protocols)
            if 'acquisition' in f:
                acq = f['acquisition']
                for key in acq.keys():
                    series = acq[key]
                    series_type = series.attrs.get('neurodata_type', b'').decode() if isinstance(series.attrs.get('neurodata_type'), bytes) else series.attrs.get('neurodata_type', '')
                    if 'VoltageClampSeries' in series_type or 'voltage' in key.lower():
                        protocols_found.add('VoltageClamp')
                    elif 'CurrentClampSeries' in series_type or 'current' in key.lower():
                        protocols_found.add('CurrentClamp')
            
            # Determine if mixed or single protocol
            protocol_type = 'mixed' if len(protocols_found) > 1 else 'single'
            
            protocol_info = {
                'protocols': list(protocols_found),
                'count': len(protocols_found),
                'file': nwb_file
            }
            
            return protocol_type, protocol_info
    
    except Exception as e:
        print(f"⚠ Warning: Could not detect protocol type: {e}")
        # Default to single protocol if detection fails
        return 'single', {'error': str(e)}


def print_header():
    """Print welcome banner"""
    print("\n" + "="*70)
    print("  ELECTROPHYSIOLOGY ANALYSIS PIPELINE")
    print("="*70)
    print("\nSupported formats:")
    print("  • ABF files (.abf) - Patch clamp electrophysiology")
    print("  • NWB files (.nwb) - Neurodata Without Borders format")
    print()


def get_file_type():
    """Prompt user to select file type"""
    print("What type of data are you analyzing?")
    print("  1) ABF files (.abf)")
    print("  2) NWB files (.nwb)")
    print()
    
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice in ["1", "2"]:
            return choice
        print("  ✗ Invalid input. Please enter 1 or 2.")


def run_abf_pipeline(no_checkpoints: bool = False):
    """Run ABF analysis pipeline"""
    print("\n" + "="*70)
    print("ABF PIPELINE")
    print("="*70)
    print("""
This pipeline will:
  1. Read all .abf files in a directory
  2. Parse ABF metadata (recording date, sweep info, etc.)
  3. Bundle sweeps into parquet format
  4. Create sweep_config.json for classification
  5. Run full analysis:
     - Resting membrane potential
     - Spike detection
     - Savitzky-Golay filtering
     - Input resistance calculation
""")
    
    # Prompt for the ABF folder directory
    while True:
        abf_dir = input("Enter the directory path containing ABF files: ").strip()
        if not abf_dir:
            print("  ✗ Please enter a path.")
            continue
        if not os.path.isdir(abf_dir):
            print(f"  ✗ Directory not found: {abf_dir}")
            continue
        # Check for ABF files (including subfolders)
        abf_files = list(Path(abf_dir).rglob("*.abf"))
        if not abf_files:
            print(f"  ✗ No .abf files found in {abf_dir} or its subfolders")
            continue
        print(f"\n✓ Found {len(abf_files)} ABF file(s) in {abf_dir} and its subfolders")
        break
    
    # Prompt for Excel metadata path
    while True:
        excel_path = input("Enter the path to the Excel metadata file: ").strip()
        if not excel_path:
            print("  ✗ Please enter a path.")
            continue
        if not os.path.isfile(excel_path):
            print(f"  ✗ File not found: {excel_path}")
            continue
        print(f"✓ Using metadata: {excel_path}")
        break
    
    print(f"\nLaunching ABF pipeline...")
    
    # Import and run zuckerman-abf functions directly to avoid subprocess stdin issues
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))
    
    try:
        # Import the module by loading the file directly (has hyphen in name)
        import importlib.util
        spec = importlib.util.spec_from_file_location("zuckerman_abf", script_dir / "zuckerman-abf.py")
        zuckerman = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(zuckerman)
        
        # Step 1: Bundle ABF files using Excel metadata (skip if bundles already exist)
        abf_path = Path(abf_dir)
        # detect existing bundle directories (manifest.json or mv_/pa_ parquet files)
        existing_bundles = []
        for d in abf_path.iterdir():
            if not d.is_dir():
                continue
            if (d / "manifest.json").exists():
                existing_bundles.append(d)
                continue
            mv_found = any(d.glob("mv_*.parquet"))
            pa_found = any(d.glob("pa_*.parquet"))
            if mv_found and pa_found:
                existing_bundles.append(d)

        if existing_bundles:
            print(f"\nFound {len(existing_bundles)} existing bundle(s). Skipping bundle creation and proceeding to analysis.")
        else:
            print(f"\n--- Step 1: Creating bundles from ABF files ---")
            zuckerman.process_mouse_folder(
                mouse_dir=abf_dir,
                excel_path=excel_path,
                out_root=abf_dir
            )
        
        # STEP 1: After bundling complete
        print(f"\n{'='*70}")
        print("✓ STEP 1: BUNDLE CREATION COMPLETE")
        print("="*70)
        print("All ABF files have been extracted and bundled with parquet files and manifest.json")
        
        # Pause/resume loop for bundling inspection (skip if no_checkpoints)
        if not no_checkpoints:
            while True:
                response = input("\nProceed to analysis? (y/n): ").strip().lower()
                if response == 'y':
                    break
                elif response == 'n':
                    print(f"\n⏸ Pipeline paused. You can inspect bundles in:")
                    print(f"  {abf_dir}")
                    print(f"\nWhen ready to resume, type 'resume':")
                    resume_input = input().strip().lower()
                    if resume_input == 'resume':
                        print("Resuming pipeline...\n")
                        break
                    else:
                        print("(Type 'resume' to continue)")
        
        # Step 2: Run analysis on all bundles
        print(f"\n{'='*70}")
        print("--- Step 2: Running analysis on bundles ---")
        print("="*70)
        from run_analysis import run_for_bundle
        
        abf_path = Path(abf_dir)
        
        # Find all ABF bundle directories (recursively look for manifest.json)
        # Filter out NWB bundles (which have "sub-" prefix directories)
        bundle_dirs = []
        for manifest_file in abf_path.rglob("manifest.json"):
            bundle_dir = manifest_file.parent
            # Skip NWB bundles (they have parent directories with "sub-" prefix)
            # NWB structure: sub-XXXX/sub-XXXX_ses-YYYY_icephys/manifest.json
            # ABF structure: YYYY_MM_DD_Mouse_X/YYYY_MM_DD_XXXX_NNN/manifest.json
            if any(p.name.startswith("sub-") for p in bundle_dir.parents):
                continue  # Skip NWB bundles
            bundle_dirs.append(bundle_dir)
        
        bundle_dirs = sorted(bundle_dirs)
        
        if not bundle_dirs:
            print(f"✗ No bundle directories found in {abf_dir}")
            print(f"  Looking for directories containing manifest.json files")
            print(f"  Make sure you've completed Step 1 (bundle creation)")
            sys.exit(1)
        
        print(f"\n📊 Found {len(bundle_dirs)} bundle(s) to analyze:")
        for bundle in bundle_dirs:
            print(f"  • {bundle.relative_to(abf_path)}")
        
        bundles_processed = 0
        for bundle in bundle_dirs:
            manifest_path = bundle / "manifest.json"
            try:
                with open(manifest_path, "r") as f:
                    man = json.load(f)
                meta = man.get("meta") or {}
                if not meta:
                    print(f"Skipping {bundle.name} (manifest has no metadata)")
                    continue
                protocol = str(meta.get("protocol", "")).lower()
                
                # Skip RAMP protocols - not suitable for spike analysis
                if "ramp" in protocol:
                    print(f"Skipping Bundle (ramp protocol): {bundle.name}")
                    continue
                
                # Only process bundles with step, intrinsic, or IV protocols
                if any(k in protocol for k in ("step", "intrinsic", "iv")):
                    print(f"\nRunning bundle {bundle.name}")
                    run_for_bundle(str(bundle), no_checkpoints=no_checkpoints)
                    bundles_processed += 1
                else:
                    print(f"Skipping Bundle (unsupported protocol '{protocol}'): {bundle.name}")
            except Exception as e:
                import traceback
                print(f"ERROR processing bundle {bundle.name}: {e}")
                traceback.print_exc()
        
        # FINAL CHECKPOINT: Entire ABF pipeline
        print(f"\n{'='*70}")
        print("✓ FINAL CHECKPOINT: ABF ANALYSIS PIPELINE COMPLETE!")
        print("="*70)
        print(f"Successfully processed {bundles_processed} bundle(s)")
        print(f"All results saved to: {abf_dir}")
        print()
        
    except Exception as e:
        print(f"✗ ERROR running ABF pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def get_parent_directory() -> Path:
    """Prompt user for parent directory containing subject folders with NWB files"""
    while True:
        parent_dir = input("\nEnter parent directory path (contains subject subfolders): ").strip()
        parent_path = Path(parent_dir)
        
        if not parent_path.exists():
            print(f"✗ Directory not found: {parent_path}")
            continue
        
        if not parent_path.is_dir():
            print(f"✗ Path is not a directory: {parent_path}")
            continue
        
        # Check if there are any NWB files in subdirectories
        nwb_files = list(parent_path.rglob("*.nwb"))
        if not nwb_files:
            print(f"✗ No .nwb files found in {parent_path} or its subdirectories")
            continue
        
        print(f"\n✓ Found {len(nwb_files)} NWB files")
        
        # Show sample of what will be processed
        print("\nSample of NWB files found:")
        for nwb in sorted(nwb_files)[:5]:
            print(f"  • {nwb.relative_to(parent_path)}")
        if len(nwb_files) > 5:
            print(f"  ... and {len(nwb_files) - 5} more")
        
        confirm = input("\nProcess all NWB files in this directory? (y/n): ").strip().lower()
        if confirm == 'y':
            return parent_path
        
        print("Please try another directory.")


def run_nwb_data_preparation(parent_dir = None, no_checkpoints: bool = False):
    """Run STEP 1: Data preparation with automatic protocol detection
    
    Args:
        parent_dir: Optional parent directory (if not provided, will prompt user)
        
    Returns:
        Path to the parent directory used (for pipeline reuse)
    """
    print("\n" + "="*70)
    print("STEP 1: DATA PREPARATION (Automatic Protocol Detection)")
    print("="*70)
    print("""
This step extracts NWB data and creates bundle directories.

For each NWB file in the directory, this will:
  • Detect if single or mixed protocol
  • Extract voltage and current traces
  • Convert units (V → mV, A → pA)
  • Save as parquet files (fast I/O)
  • Create manifest.json with metadata

Single protocol: Uses standard process_human_data.py
Mixed protocol:  Uses process_human_data_mixed_protocol.py with per-sweep rates
""")
    
    # Get parent directory containing subject folders with NWB files
    if parent_dir is None:
        parent_dir = get_parent_directory()
    else:
        parent_dir = Path(parent_dir)
    
    # Find all NWB files
    nwb_files = sorted(parent_dir.rglob("*.nwb"))
    print(f"\n📊 Found {len(nwb_files)} NWB file(s) to process")
    
    # Analyze protocol type for each file
    protocol_analysis = {}
    print(f"\n📊 Analyzing NWB file structures...")
    
    for nwb_file in nwb_files:
        filename = nwb_file.stem
        subject_id = filename.split('_')[0] if '_' in filename else 'unknown'
        
        protocol_type, protocol_info = detect_nwb_protocol_type(str(nwb_file))
        
        if protocol_info is None:
            protocol_type = 'single'
            protocol_info = {}
        
        protocol_analysis[str(nwb_file)] = {
            'subject_id': subject_id,
            'type': protocol_type,
            'info': protocol_info
        }
        
        protocols_str = ', '.join(protocol_info.get('protocols', ['unknown']))
        print(f"  • {filename}: {protocol_type.upper()} ({protocols_str})")
    
    # Call appropriate processing script(s)
    script_dir = Path(__file__).parent
    
    try:
        print(f"\n🔄 Starting extraction process...")
        
        # Validate template file ONCE before processing any files
        template_path = script_dir / 'ePhys_log_sheet.xlsx'
        if not template_path.is_file():
            print(f"\n⚠ Template file not found: {template_path}")
            if no_checkpoints:
                print("Cannot proceed without metadata template in no-checkpoints mode.")
                sys.exit(1)
            print("Please provide the path to the Excel metadata template:")
            template_path_input = input("Path to Excel metadata template: ").strip()
            template_path = Path(template_path_input).expanduser()

            # Validate the user-provided path
            if not template_path.is_file():
                print(f"\n✗ ERROR: Template file not found: {template_path}")
                print("Cannot proceed without metadata template.")
                sys.exit(1)

            print(f"✓ Using template: {template_path}\n")
        
        # Separate files by protocol type
        single_protocol_files = []
        mixed_protocol_files = []
        
        for nwb_file_path, analysis in protocol_analysis.items():
            if analysis['type'] == 'mixed':
                mixed_protocol_files.append(nwb_file_path)
            else:
                single_protocol_files.append(nwb_file_path)
        
        # Process mixed protocol files (one at a time)
        if mixed_protocol_files:
            print(f"\n📊 Processing {len(mixed_protocol_files)} mixed protocol file(s)...\n")
            
            # Track cell count per subject to pass to script
            subject_cell_counts = {}
            
            for nwb_file in mixed_protocol_files:
                # Extract subject ID and get count (starts at 0)
                subject_id = Path(nwb_file).stem.split('_')[0]  # e.g., "sub-1000610030"
                cell_count = subject_cell_counts.get(subject_id, 0)
                subject_cell_counts[subject_id] = cell_count + 1  # Increment for next file
                
                print(f"\n{'='*70}")
                print(f"Processing: {Path(nwb_file).name}")
                print(f"Protocol: MIXED")
                print(f"Cell: {subject_id}_{cell_count}")
                print('='*70)
                print(f"\U0001f504 Mixed protocol detected - Using enhanced extraction...")
                print(f"   Launching process_human_data_mixed_protocol.py...\n")
                
                # Use generic mixed protocol script for all mixed protocol files
                mixed_script = script_dir / 'process_human_data_mixed_protocol.py'
                if mixed_script.exists():
                    # Pass: parent_dir, log_output_dir, template_path, nwb_file, cell_count
                    result = subprocess.run(
                        [sys.executable, str(mixed_script), 
                         str(parent_dir), str(parent_dir), str(template_path),
                         str(nwb_file), str(cell_count)],
                        check=False
                    )
                    if result.returncode != 0:
                        print(f"\n⚠ Processing script exited with code {result.returncode}")
                        print(f"  (Continuing with next file if available)")
                else:
                    # Warn if script not found
                    print(f"   ⚠ Mixed protocol script not found: {mixed_script}")
                    print(f"   Skipping this file.\n")
                
                # Force garbage collection after each file to free memory
                gc.collect()
        
        # Process single protocol files (all at once)
        if single_protocol_files:
            print(f"\n📊 Processing {len(single_protocol_files)} single protocol file(s)...\n")
            print(f"\n{'='*70}")
            print(f"Processing: {len(single_protocol_files)} single protocol files")
            print(f"Protocol: SINGLE")
            print('='*70)
            print(f"\U0001f504 Single protocol detected - Using standard extraction...")
            print(f"   Launching process_human_data.py...\n")
            result = subprocess.run(
                [sys.executable, str(script_dir / 'process_human_data.py'), 
                 str(parent_dir), str(parent_dir), str(template_path)],
                check=False
            )
            if result.returncode != 0:
                print(f"\n⚠ Processing script exited with code {result.returncode}")
        
        print(f"\n{'='*70}")
        print(f"✓ Data preparation complete!")
        print('='*70)
        
        # Return parent directory for reuse in pipeline
        return parent_dir
            
    except Exception as e:
        print(f"✗ ERROR running data preparation: {e}")
        sys.exit(1)


def run_nwb_analysis(parent_dir = None, no_checkpoints: bool = False):
    """Run STEP 2: Analysis on bundle directories
    
    Args:
        parent_dir: Optional parent directory (if not provided, will prompt user)
    """
    print("\n" + "="*70)
    print("STEP 2: ANALYSIS & CLASSIFICATION (bundle_analyzer.py)")
    print("="*70)
    print("""
This step analyzes all bundles and runs spike detection.

For each bundle, this will:
  • Load parquet files (voltage + current)
  • Classify sweeps (kept vs dropped)
  • Create sweep_config.json with metadata
  • Run full analysis:
    - Resting membrane potential
    - Spike detection
    - Savitzky-Goyal filtering
    - Input resistance calculation
""")
    
    # Get parent directory containing bundle directories
    if parent_dir is None:
        parent_dir = get_parent_directory()
    
    parent_path = Path(parent_dir)
    
    # Look for directories that contain manifest.json (they are bundle directories)
    bundle_dirs = []
    for subfolder in parent_path.rglob("manifest.json"):
        # Look for bundle directories inside each subject subfolder
        bundle_dir = subfolder.parent
        if any(p.name.startswith("sub-") for p in bundle_dir.parents):
                bundle_dirs.append(bundle_dir)
        # for bundle_candidate in subfolder.glob("*"):
        #     if bundle_candidate.is_dir() and (bundle_candidate / "manifest.json").exists():
        #         bundle_dirs.append(bundle_candidate)
    
    bundle_dirs = sorted(bundle_dirs)
    
    if not bundle_dirs:
        print(f"✗ ERROR: No bundle directories found in {parent_path}")
        print(f"  Looking for directories containing manifest.json files")
        print(f"  Make sure you've completed STEP 1 (data preparation)")
        sys.exit(1)
    
    print(f"\n📊 Found {len(bundle_dirs)} bundle(s) to analyze:")
    for bundle in bundle_dirs:
        print(f"  • {bundle.relative_to(parent_path)}")
    
    # Ask about optional flags
    print("\nAnalysis options:")
    print("  1) Run full analysis with plots (default)")
    print("  2) Only create sweep_config.json (skip spike detection)")

    if no_checkpoints:
        analysis_choice = "1"
        print("  [no-checkpoints] Auto-selecting option 1 (full analysis)")
    else:
        while True:
            analysis_choice = input("Enter 1 or 2: ").strip()
            if analysis_choice in ["1", "2"]:
                break
            print("  ✗ Invalid input. Please enter 1 or 2.")
    
    # Ask about resuming from a specific bundle
    start_idx = 0
    if len(bundle_dirs) > 1 and not no_checkpoints:
        print(f"\nResume options:")
        print(f"  Press ENTER to start from the beginning")
        print(f"  Or enter a bundle number (1-{len(bundle_dirs)}) to resume from:")
        for i, b in enumerate(bundle_dirs, 1):
            print(f"    {i}) {b.name}")
        resume_input = input("Resume from (ENTER=start): ").strip()
        if resume_input.isdigit():
            start_idx = max(0, int(resume_input) - 1)
            if start_idx > 0:
                print(f"  \u2192 Skipping first {start_idx} bundle(s), starting from #{start_idx + 1}")
    
    try:
        from bundle_analyzer import main as analyzer_main
        
        # Process each bundle (optionally skipping already-completed ones)
        for idx, bundle_path in enumerate(bundle_dirs, 1):
            if idx - 1 < start_idx:
                print(f"\n[{idx}/{len(bundle_dirs)}] Skipping (already done): {bundle_path.name}")
                continue
            
            print(f"\n{'='*70}")
            print(f"[{idx}/{len(bundle_dirs)}] Analyzing: {bundle_path.relative_to(parent_path)}")
            print('='*70)
            
            # Set up sys.argv for argparse
            argv = ["bundle_analyzer.py", str(bundle_path)]
            
            if analysis_choice == "2":
                argv.append("--skip-analysis")

            if no_checkpoints:
                argv.append("--no-checkpoints")
            
            sys.argv = argv
            
            try:
                analyzer_main()
                print(f"✓ Analysis complete for {bundle_path.name}")
            except Exception as e:
                print(f"⚠ ERROR analyzing {bundle_path.name}: {e}")
                print(f"  Continuing with next bundle...")
                continue
        
        print(f"\n{'='*70}")
        print(f"✓ Analysis complete!")
        print('='*70)
            
    except ImportError as e:
        print(f"✗ ERROR: Could not import bundle_analyzer.py")
        print(f"  Make sure bundle_analyzer.py is in the current directory")
        print(f"  Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ ERROR running analysis: {e}")
        sys.exit(1)


def run_nwb_pipeline(no_checkpoints: bool = False):
    """Run full NWB analysis pipeline (both steps)"""
    print("\n" + "="*70)
    print("NWB ANALYSIS PIPELINE")
    print("="*70)
    print("""
This pipeline has two steps:

STEP 1: DATA PREPARATION (process_human_data.py)
  NWB files → Extract traces → Parquet files + Bundles

STEP 2: ANALYSIS & CLASSIFICATION (bundle_analyzer.py)
  Bundle directory → Classify sweeps → Run spike detection

Workflow options:
  1) Full pipeline (both steps) - for new NWB files
  2) Data preparation only (STEP 1) - prepare bundles first
  3) Analysis only (STEP 2) - for existing bundles
""")
    
    while True:
        pipeline_choice = input("Enter 1, 2, or 3: ").strip()
        if pipeline_choice in ["1", "2", "3"]:
            break
        print("  ✗ Invalid input. Please enter 1, 2, or 3.")
    
    if pipeline_choice == "1":
        # Full pipeline: both steps
        print("\n" + "="*70)
        print("FULL PIPELINE: DATA PREP + ANALYSIS")
        print("="*70)
        print("""
This will run both steps in sequence:
  1. Extract NWB files to parquets
  2. Run analysis on all created bundles

Note: You will need to provide paths for:
  - Parent directory with NWB files
  - Output directory for bundles
  - Metadata Excel template
""")
        
        # Run data preparation and capture the parent directory
        parent_dir = run_nwb_data_preparation(no_checkpoints=no_checkpoints)
        
        # Automatically run analysis on all bundles
        print("\n" + "="*70)
        print("Data preparation complete!")
        print("="*70)
        
        # Skip checkpoint if no_checkpoints flag is set
        if no_checkpoints:
            analyze_now = "y"  # Auto-proceed
        else:
            analyze_now = input("\nRun analysis on all bundles now? (y/n): ").strip().lower()
        
        if analyze_now == "y":
            # Pass the parent directory from data prep to avoid re-prompting
            run_nwb_analysis(parent_dir, no_checkpoints=no_checkpoints)
            print("\n" + "="*70)
            print("✓ PIPELINE COMPLETE")
            print("="*70)
            print()
        else:
            print("\n" + "="*70)
            print("✓ DATA PREPARATION COMPLETE")
            print("="*70)
            print("\nNext step:")
            print("  Run analysis: python main.py  (select option 2 → NWB → Analysis only)")
            print()
    
    elif pipeline_choice == "2":
        # Data preparation only
        run_nwb_data_preparation(no_checkpoints=no_checkpoints)
        
        print("\n" + "="*70)
        print("✓ DATA PREPARATION COMPLETE")
        print("="*70)
        print("\nNext step:")
        print("  Run analysis: python main.py  (select option 2 → NWB → Analysis only)")
        print()
    
    else:  # pipeline_choice == "3"
        # Analysis only
        run_nwb_analysis(no_checkpoints=no_checkpoints)
        print("\n" + "="*70)
        print("✓ ANALYSIS COMPLETE")
        print("="*70)
        print()


def main():
    """Main entry point"""
    print_header()
    
    # Check for command-line arguments
    no_checkpoints = '--no-checkpoints' in sys.argv
    if no_checkpoints:
        print("ℹ Running in non-interactive mode (skipping checkpoints)")
        sys.argv.remove('--no-checkpoints')  # Remove from sys.argv to avoid breaking other parsers
    
    file_type = get_file_type()
    
    if file_type == "1":
        run_abf_pipeline(no_checkpoints=no_checkpoints)
        print("\n" + "="*70)
        print("✓ PIPELINE COMPLETE")
        print("="*70)
        print()
    else:  # file_type == "2"
        run_nwb_pipeline(no_checkpoints=no_checkpoints)
        # Success message printed by run_nwb_pipeline()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
