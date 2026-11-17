# Kaggle pilot: 2D, fold 0, 100 epochs

## Budget and stages

1. `hichandut/brain-mri-prepare-cpu`: CPU-only notebook downloads the official MSD archive,
   verifies MD5, checks labels/counts, converts training cases, preprocesses **2D only** and writes five fixed folds.
   It excludes the 266 unlabeled test images. Raw/temporary copies are removed before final output.
2. `hichandut/brain-mri-2d-100ep`: takes that completed notebook output as an input;
   uses upstream **nnUNetTrainer_100epochs**, fold 0, and a two-hour training budget.
   Full-volume validation is deliberately separate and not run automatically.

The initial CLI quota check on 2026-11-17 reported **1.38 h used / 28.62 h remaining**.
This is a snapshot, not a promise of current remaining quota. Check again with `make kaggle-quota`.

The GPU notebook requests **one L4** if available for the account. If the API explicitly rejects that hardware,
the launcher can fall back to Kaggle's **T4 ×2** option. The trainer uses CUDA device 0 only; the second card
is not used for another experiment. Check the saved launcher state for the actual selected accelerator.
Do not assume that masking one GPU changes the platform's quota accounting.

The training budget stops at an epoch boundary, reserving the observed epoch duration and a checkpoint margin.
A particularly slow first epoch can exceed that soft training budget. The notebook has a separate **3-hour hard
runtime limit**, including installation, input copying and finalization; it is not an exact two-hour quota cap.

## Commands

```bash
make kaggle-quota
make kaggle-build       # regenerate notebooks from scripts
make kaggle-prepare     # submits CPU preprocessing; do not repeat while a run is active
make kaggle-status
# After CPU stage completes, either use the one-shot launcher:
python kaggle/launch_after_prepare.py
# Or submit manually (do not also run the launcher):
make kaggle-train
```

The launcher polls CPU status, checks quota immediately before GPU submission and refuses a duplicate
submission once its local state is `gpu_submitted`. It never silently schedules additional folds or weeks of training.
`artifacts/kaggle-launcher.json` and `.log` contain local launch state and logs.

## Training behavior

- 100-epoch learning-rate schedule from the start; 250 training steps and 50 sampled validation steps per epoch.
- Checkpoint every five epochs plus the best EMA pseudo-Dice checkpoint; writes are atomic.
- At a budget stop, write `checkpoint_latest.pth` with the correct next-epoch index.
- Report median epoch time (excluding the first two warmup epochs) and estimated time remaining.
- `completed_100_epochs` means training finished; `paused_budget_checkpoint_saved` means a partial run.
- Log pseudo-Dice is a patch-based training monitor, **not** the final full-volume Dice metric.

`train_budget.py --resume /path/to/checkpoint_latest.pth` resumes with the same 100-epoch schedule.
Use the same plans, dataset metadata, fold split and trainer. For a later Kaggle session, attach the earlier
results as an input, restore the matching results folder, and supply that checkpoint explicitly. The current
first-run notebook does not automatically schedule resume sessions.

## Outputs

CPU: `preprocessed/` and `prepare-report.json` (no GPU quota).
GPU: `results/Dataset501_BrainTumour/nnUNetTrainer_100epochs__nnUNetPlans__2d/fold_0/`,
`train-report.json`, plans and dataset metadata. Partial checkpoints retain optimizer/scaler state for continuation.
The class is part of pinned nnU-Net 2.8.1; a custom trainer installation is not needed for ordinary inference.

```bash
make kaggle-output
```

No real model performance or training-time benchmark is claimed until the job produces measurements.
Kaggle notebooks are private. Dataset attribution and CC-BY-SA 4.0 remain attached to CPU output metadata.
