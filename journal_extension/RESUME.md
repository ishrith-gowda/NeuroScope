# disaster-recovery / resume runbook (sashimi ext a + ext c)

the original patchnce checkpoints were lost once when a chameleon lease lapsed.
this runbook + the automatic backup loop ensure we can always pick up where we
left off. **nothing critical lives only on the ephemeral node.**

## what is backed up, and where

| artifact | location (durable) | mechanism |
|----------|--------------------|-----------|
| code (scripts, scaffolds, configs) | github.com/ishrith-gowda/NeuroScope (main) | git commit + push |
| preprocessed data (9.3G) + base ckpt | local: `preprocessed/`, `final_trained_model/` | source of truth (never only on node) |
| in-progress checkpoints + logs + tb | local: `cluster_backup/` | **background rsync loop, node -> local, every 15 min** |
| per-run provenance | inside each run dir: `config.json`, `training_history.json`, `checkpoints/` | written by the trainer |

each checkpoint stores full resume state: model + opt_G/opt_D + scheduler_G/D +
scaler_G/D + history + global_step + best_val_ssim + epoch.

## node facts
- host: `cc@192.5.86.245` (compute_gigaio A100 80GB, CC-Ubuntu22.04-CUDA)
- key: `~/.ssh/neuroscope-key`
- workdir: `~/neuroscope/{code,preprocessed,checkpoints,experiments,logs,runs}`
- env: `~/neuroscope/code/.venv` (uv, py3.11, torch 2.6+cu124)
- experiments: `~/neuroscope/experiments/ext_a/ext_a_lambda{0.5,0.0,1.0,2.0,0.1}/checkpoints/`

## if the node dies / lease expires -> recover on a fresh node
1. provision compute_gigaio with `CC-Ubuntu22.04-CUDA`; confirm `~/.ssh/neuroscope-key` auth + note the new IP.
2. env:
   ```
   curl -LsSf https://astral.sh/uv/install.sh | sh
   mkdir -p ~/neuroscope/{preprocessed,checkpoints,experiments,logs,runs}
   # rsync code + data + base ckpt + last checkpoints back UP from local:
   rsync -a code/         cc@NEW_IP:~/neuroscope/code/
   rsync -a preprocessed/brats preprocessed/upenn cc@NEW_IP:~/neuroscope/preprocessed/
   rsync -a final_trained_model/checkpoints/checkpoint_best.pth cc@NEW_IP:~/neuroscope/checkpoints/base_sa_cyclegan_best.pth
   rsync -a cluster_backup/experiments/ cc@NEW_IP:~/neuroscope/experiments/   # RESTORES in-progress checkpoints
   cd ~/neuroscope/code && uv venv --python 3.11 .venv && source .venv/bin/activate
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   uv pip install numpy scipy scikit-image nibabel SimpleITK pyyaml tqdm tensorboard scikit-learn matplotlib pandas h5py torchmetrics monai
   ```
3. resume the interrupted arm (full state restored):
   ```
   python journal_extension/scripts/finetune_ext_a_from_base.py \
     --resume ~/neuroscope/experiments/ext_a/ext_a_lambda<L>/<exp>/checkpoints/checkpoint_latest.pth \
     --brats_dir ~/neuroscope/preprocessed/brats --upenn_dir ~/neuroscope/preprocessed/upenn \
     --output_dir ~/neuroscope/experiments/ext_a --experiment_name ext_a_lambda<L> \
     --lambda_nce <L> --epochs 30
   ```
   then resume any not-yet-started arms with `--base_checkpoint` as usual.
4. restart the backup loop (background rsync node -> local, 15 min).

## lease hygiene
- renew the chameleon lease BEFORE expiry (check the reservation end time). the
  backup loop protects against unexpected death; renewal protects against
  scheduled expiry. set a calendar reminder for the lease end.
