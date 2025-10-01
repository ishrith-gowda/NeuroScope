import subprocess
import sys
import shutil
import json
from pathlib import Path
import argparse
import logging

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
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"Stage {entry['name']} failed: {result.stderr[:300]}")
        return False
    logging.debug(result.stdout[-500:])
    return True

def check_outputs(entry):
    # Heuristic checks
    for out_name in entry['required_outputs']:
        # Determine location: JSON results go to preprocessed_dir root; images validated per subject
        if out_name.endswith('.json'):
            candidate = PATHS['preprocessed_dir'] / out_name
            if not candidate.exists():
                return False
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
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format='[%(levelname)s] %(message)s')


def main():
    args = parse_args()
    setup_logging(args.verbose)
    splits = [s.strip() for s in args.splits.split(',') if s.strip()]

    logging.info(f"Starting preprocessing pipeline for splits: {splits}")

    base_args = [f"--splits={','.join(splits)}"]
    if args.force:
        base_args.append('--overwrite')

    summary = []

    for stage in STAGES:
        already_ok = check_outputs(stage)
        if already_ok and not args.force:
            logging.info(f"Skipping {stage['name']} (outputs present)")
            summary.append({stage['name']: 'skipped_present'})
            continue
        success = run_stage(stage, base_args, dry_run=args.dry_run)
        if not success:
            summary.append({stage['name']: 'failed'})
            if args.stop_on_fail:
                break
        else:
            # Re-check outputs
            produced = check_outputs(stage)
            summary.append({stage['name']: 'ok' if produced else 'warning_outputs_missing'})
            if not produced and args.stop_on_fail:
                break

    report_path = PATHS['preprocessed_dir'] / 'pipeline_run_summary.json'
    try:
        with open(report_path, 'w') as f:
            json.dump({'run_summary': summary}, f, indent=2)
        logging.info(f"Pipeline run summary written: {report_path}")
    except Exception as e:
        logging.warning(f"Could not write summary: {e}")

    logging.info('Pipeline complete')

if __name__ == '__main__':
    main()
