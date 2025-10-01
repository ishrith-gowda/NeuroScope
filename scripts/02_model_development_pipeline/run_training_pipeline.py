import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

# Use PATHS for defaults
HERE = Path(__file__).resolve().parent
PREP_DIR = HERE.parent / '01_data_preparation_pipeline'
sys.path.insert(0, str(PREP_DIR))
import neuroscope_preprocessing_config as npc  # type: ignore
PATHS = npc.PATHS


STAGES = [
    {
        'name': 'prepare_manifest',
        'script': '01_prepare_training_manifest.py',
        'args': [],
        'outputs': [HERE / 'neuroscope_training_manifest.json']
    },
    {
        'name': 'comprehensive_validation',
        'script': '06_comprehensive_pipeline_validation.py',
        'args': ['--verbose'],
        'outputs': [HERE / 'pipeline_validation_report.json']
    },
    {
        'name': 'dataloader_smoke_test',
        'script': '02_dataloader_smoke_test.py',
        'args': [
            f"--preprocessed_dir={PATHS['preprocessed_dir']}",
            f"--metadata_json={PATHS['metadata_splits']}"
        ],
        'outputs': []
    },
    {
        'name': 'train_cyclegan',
        'script': '03_train_cyclegan_entry.py',
        'args': [],
        'outputs': [PATHS['checkpoints_dir']]
    },
    {
        'name': 'evaluate',
        'script': '04_evaluate_cyclegan.py',
        'args': [],  # filled at runtime using latest checkpoint
        'outputs': [PATHS['figures_dir']]
    },
]


def setup_logging(verbose: bool):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format='[%(levelname)s] %(message)s')


def run_py(script_path: Path, args: list) -> int:
    cmd = [sys.executable, str(script_path)] + list(args)
    logging.info('→ %s', ' '.join(str(x) for x in cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logging.error('stderr: %s', res.stderr[-600:])
    else:
        logging.debug(res.stdout[-600:])
    return res.returncode


def latest_checkpoint(ckpt_dir: Path, prefix: str = 'G_A2B_', suffix: str = '.pth') -> Path | None:
    if not ckpt_dir.exists():
        return None
    candidates = sorted(ckpt_dir.glob(f"{prefix}*{suffix}"))
    return candidates[-1] if candidates else None


def parse_args():
    ap = argparse.ArgumentParser(description='Run modular CycleGAN training pipeline')
    ap.add_argument('--force', action='store_true', help='Re-run all stages')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--skip-validation', action='store_true', help='Skip comprehensive validation')
    ap.add_argument('--skip-train', action='store_true')
    ap.add_argument('--skip-eval', action='store_true')
    return ap.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)

    # 1) Prepare manifest
    stage = STAGES[0]
    script = HERE / stage['script']
    if args.force or not all(Path(p).exists() for p in stage['outputs']):
        rc = run_py(script, stage['args'])
        if rc != 0:
            sys.exit(1)
    else:
        logging.info('skip prepare_manifest (outputs present)')

    # 2) Comprehensive validation (optional)
    if not args.skip_validation:
        stage = STAGES[1]
        if args.force or not all(Path(p).exists() for p in stage['outputs']):
            rc = run_py(HERE / stage['script'], stage['args'])
            if rc != 0:
                logging.error("Comprehensive validation failed - check validation report")
                sys.exit(1)
        else:
            logging.info('skip comprehensive_validation (outputs present)')
    else:
        logging.info('skip comprehensive_validation by flag')

    # 3) Smoke test
    stage = STAGES[2]
    rc = run_py(HERE / stage['script'], stage['args'])
    if rc != 0:
        sys.exit(1)

    # 4) Train
    if not args.skip_train:
        stage = STAGES[3]
        rc = run_py(HERE / stage['script'], stage['args'])
        if rc != 0:
            sys.exit(1)
    else:
        logging.info('skip train stage by flag')

    # 5) Evaluate latest
    if not args.skip_eval:
        ckpt = latest_checkpoint(Path(PATHS['checkpoints_dir']))
        if ckpt is None:
            logging.warning('no checkpoint found for evaluation')
            return
        eval_args = [
            f"--generator_ckpt={ckpt}",
            f"--data_root={PATHS['preprocessed_dir']}",
            f"--meta_json={PATHS['metadata_splits']}",
            f"--output_dir={PATHS['figures_dir']}",
            '--split=val'
        ]
        rc = run_py(HERE / '04_evaluate_cyclegan.py', eval_args)
        if rc != 0:
            sys.exit(1)
    else:
        logging.info('skip evaluation by flag')

    logging.info('Training pipeline complete')


if __name__ == '__main__':
    main()
