import subprocess
import sys
import shutil
import json
import time
from pathlib import Path
import argparse
import logging
import threading

from neuroscope_preprocessing_config import PATHS

STAGES = [
    {
        'name': 'fast_intensity_normalization',
        'script': '01_fast_intensity_normalization_neuroscope.py',
        'required_outputs': ['t1.nii.gz'],
        'args': []
    },
    {
        'name': 'bias_assessment_pre_n4',
        'script': '05_comprehensive_intensity_bias_assessment_neuroscope.py',
        'required_outputs': ['slice_bias_assessment.json'],
        'args': []
    },
    {
        'name': 'n4_bias_correction',
        'script': '06_n4_bias_correction_neuroscope.py',
        'required_outputs': ['n4_correction_results_improved_v2.json'],
        'args': []
    },
    {
        'name': 'n4_effectiveness',
        'script': '07_assess_n4_correction_effectiveness_neuroscope.py',
        'required_outputs': ['n4_effectiveness_assessment.json'],
        'args': []
    },
    {
        'name': 'n4_diagnostics',
        'script': '08_diagnose_n4_issues_neuroscope.py',
        'required_outputs': ['n4_diagnostic_analysis.json'],
        'args': []
    },
    {
        'name': 'pipeline_verification',
        'script': '09_verify_preprocessing_completeness_neuroscope.py',
        'required_outputs': ['neuroscope_pipeline_verification_results.json'],
        'args': []
    }
]

SCRIPT_DIR = Path(__file__).parent


def run_stage(entry, base_args, dry_run=False):
    script_path = SCRIPT_DIR / entry['script']
    if not script_path.exists():
        logging.error(f"Script not found: {script_path}")
        return False
    cmd = [sys.executable, str(script_path)] + base_args + entry.get('args', [])
    logging.info(f"Running stage {entry['name']} -> {' '.join(cmd)}")
    if dry_run:
        return True
    
    # Run process with real-time output
    logging.info(f"=== STARTING STAGE: {entry['name']} ===")
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # Print output in real time with timestamps
    import time
    last_output_time = time.time()
    
    def print_output(stream, prefix):
        nonlocal last_output_time
        for line in stream:
            print(f"[{prefix}] {line.rstrip()}")
            last_output_time = time.time()
            
    import threading
    stdout_thread = threading.Thread(target=print_output, args=(process.stdout, entry['name']))
    stderr_thread = threading.Thread(target=print_output, args=(process.stderr, f"{entry['name']}-ERROR"))
    
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    stdout_thread.start()
    stderr_thread.start()
    
    # Add progress indicator when no output for a while
    while process.poll() is None:
        time.sleep(5)  # Check every 5 seconds
        current_time = time.time()
        if current_time - last_output_time > 15:  # No output for 15 seconds
            print(f"[{entry['name']}] Still running... (no output for {int(current_time - last_output_time)} seconds)")
            last_output_time = current_time
    
    # Wait for threads to complete
    stdout_thread.join()
    stderr_thread.join()
    
    exitcode = process.returncode
    logging.info(f"=== COMPLETED STAGE: {entry['name']} (exit code: {exitcode}) ===")
    
    if exitcode != 0:
        logging.error(f"Stage {entry['name']} failed with exit code {exitcode}")
        return False
    return True

def check_outputs(entry):
    # Heuristic checks
    missing = []
    for out_name in entry['required_outputs']:
        # Special case for slice_bias_assessment.json which is stored in scripts dir
        if out_name == 'slice_bias_assessment.json':
            candidate = PATHS['slice_bias_assessment']
        # Special case for n4_diagnostic_analysis.json which doesn't exist in PATHS
        elif out_name == 'n4_diagnostic_analysis.json':
            candidate = PATHS['preprocessed_dir'] / out_name
        # Other JSON results go to preprocessed_dir root; images validated per subject
        elif out_name.endswith('.json'):
            candidate = PATHS['preprocessed_dir'] / out_name
        else:
            candidate = PATHS['preprocessed_dir'] / out_name
            
        if not candidate.exists():
            missing.append(str(candidate))
            logging.warning(f"Missing expected output: {candidate}")
    
    if missing:
        logging.warning(f"Stage {entry['name']} missing {len(missing)}/{len(entry['required_outputs'])} expected outputs")
        return False
        
    if entry['required_outputs']:
        logging.info(f"Stage {entry['name']} verified {len(entry['required_outputs'])} expected outputs ✓")
    return True


def parse_args():
    ap = argparse.ArgumentParser(description='Run full NeuroScope preprocessing pipeline')
    ap.add_argument('--splits', type=str, default='train,val', help='Splits for applicable stages')
    ap.add_argument('--force', action='store_true', help='Re-run all stages even if outputs exist')
    ap.add_argument('--dry-run', action='store_true', help='Show planned actions without executing')
    ap.add_argument('--stop-on-fail', action='store_true', help='Stop pipeline at first failed stage')
    ap.add_argument('--verbose', action='store_true')
    return ap.parse_args()


def setup_logging(verbose: bool):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO, 
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    args = parse_args()
    setup_logging(args.verbose)
    splits = [s.strip() for s in args.splits.split(',') if s.strip()]

    logging.info(f"Starting preprocessing pipeline for splits: {splits}")
    logging.info(f"Configuration: force={args.force}, dry_run={args.dry_run}, stop_on_fail={args.stop_on_fail}")
    
    # Display the pipeline stages for clarity
    logging.info("Pipeline stages to run:")
    for i, stage in enumerate(STAGES):
        logging.info(f"  {i+1}. {stage['name']} (script: {stage['script']})")

    base_args = [f"--splits={','.join(splits)}"]
    if args.force:
        base_args.append('--overwrite')

    summary = []

    total_stages = len(STAGES)
    for idx, stage in enumerate(STAGES):
        stage_num = idx + 1
        logging.info(f"\n{'='*80}\nSTAGE {stage_num}/{total_stages}: {stage['name']}\n{'='*80}")
        
        already_ok = check_outputs(stage)
        if already_ok and not args.force:
            logging.info(f"Skipping stage {stage_num}/{total_stages} '{stage['name']}' (outputs already present)")
            summary.append({stage['name']: 'skipped_present'})
            continue
            
        logging.info(f"Running stage {stage_num}/{total_stages} '{stage['name']}'")
        start_time = time.time()  # Add time tracking
        
        success = run_stage(stage, base_args, dry_run=args.dry_run)
        
        # Calculate elapsed time
        elapsed = time.time() - start_time
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        
        if not success:
            logging.error(f"Stage {stage_num}/{total_stages} '{stage['name']}' FAILED after {time_str}")
            summary.append({stage['name']: 'failed'})
            if args.stop_on_fail:
                logging.error(f"Stopping pipeline due to failure (--stop-on-fail)")
                break
        else:
            # Re-check outputs
            produced = check_outputs(stage)
            status = 'ok' if produced else 'warning_outputs_missing'
            summary.append({stage['name']: status})
            
            if produced:
                logging.info(f"Stage {stage_num}/{total_stages} '{stage['name']}' completed successfully in {time_str}")
            else:
                logging.warning(f"Stage {stage_num}/{total_stages} '{stage['name']}' ran without errors but expected outputs not found. Elapsed: {time_str}")
                
            if not produced and args.stop_on_fail:
                logging.error(f"Stopping pipeline due to missing outputs (--stop-on-fail)")
                break

    # Create final report
    report_path = PATHS['preprocessed_dir'] / 'pipeline_run_summary.json'
    try:
        with open(report_path, 'w') as f:
            json.dump({'run_summary': summary}, f, indent=2)
        logging.info(f"Pipeline run summary written: {report_path}")
    except Exception as e:
        logging.warning(f"Could not write summary: {e}")

    # Print a nice summary table
    logging.info(f"\n{'='*80}")
    logging.info("PIPELINE EXECUTION SUMMARY")
    logging.info(f"{'='*80}")
    logging.info(f"{'Stage Name':<30} {'Status':<20}")
    logging.info(f"{'-'*30} {'-'*20}")
    
    status_count = {"ok": 0, "failed": 0, "skipped_present": 0, "warning_outputs_missing": 0}
    
    for item in summary:
        for stage_name, status in item.items():
            status_count[status] = status_count.get(status, 0) + 1
            # Format status for display
            display_status = {
                "ok": "✅ SUCCESS", 
                "failed": "❌ FAILED",
                "skipped_present": "⏩ SKIPPED (present)", 
                "warning_outputs_missing": "⚠️ WARNING (missing outputs)"
            }.get(status, status)
            logging.info(f"{stage_name:<30} {display_status:<20}")
    
    logging.info(f"{'-'*80}")
    logging.info(f"SUCCESS: {status_count['ok']} | FAILED: {status_count['failed']} | " 
                f"SKIPPED: {status_count['skipped_present']} | WARNINGS: {status_count['warning_outputs_missing']}")
    logging.info(f"{'='*80}")
    
    if status_count['failed'] > 0:
        logging.warning('Pipeline completed with some failures')
    else:
        logging.info('Pipeline completed successfully')

if __name__ == '__main__':
    main()
