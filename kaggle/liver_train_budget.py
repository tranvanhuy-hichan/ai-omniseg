"""100-epoch nnU-Net pilot (Liver, 3d_lowres) with benchmark reporting and an epoch-boundary time budget.

Uses the upstream nnUNetTrainer_100epochs class so checkpoints remain compatible with
ordinary nnU-Net inference. A partial run is never labeled a completed 100-epoch model.
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path


class BudgetReached(Exception):
    pass


def estimate(durations, completed, target):
    steady = durations[2:] if len(durations) > 2 else durations
    seconds = statistics.median(steady) if steady else None
    return {'seconds_per_epoch_median': seconds,
            'estimated_remaining_hours': (target-completed)*seconds/3600 if seconds else None}


def save_for_next_epoch(trainer, path):
    """Save a boundary checkpoint with nnU-Net's next-epoch convention."""
    next_epoch = trainer.current_epoch
    trainer.current_epoch = next_epoch - 1
    try:
        trainer.save_checkpoint(str(path))
    finally:
        trainer.current_epoch = next_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hours', type=float, default=2.)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if not 0 < args.hours <= 8: parser.error('Budget must be in (0, 8] hours')
    if args.fold not in range(5): parser.error('Fold must be 0..4')
    os.environ.setdefault('nnUNet_compile', 'false')
    os.environ.setdefault('nnUNet_n_proc_DA', '2')
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    import torch
    from importlib.metadata import version
    from nnunetv2.run.run_training import get_trainer_from_args
    if version('nnunetv2') != '2.8.1': raise RuntimeError('Expected nnunetv2==2.8.1')
    if not torch.cuda.is_available(): raise RuntimeError('No GPU available; refusing CPU training')
    trainer = get_trainer_from_args('502', '3d_lowres', args.fold,
                                   trainer_name='nnUNetTrainer_100epochs',
                                   continue_training=bool(args.resume), device=torch.device('cuda', 0))
    if args.resume:
        if not args.resume.is_file(): raise FileNotFoundError(args.resume)
        trainer.load_checkpoint(str(args.resume))
    trainer.save_every = 5
    output = Path(trainer.output_folder)
    output.mkdir(parents=True, exist_ok=True)
    original_save = trainer.save_checkpoint
    def atomic_save(filename):
        target = Path(filename); temporary = target.with_suffix(target.suffix+'.tmp')
        original_save(str(temporary))
        temporary.replace(target)
    trainer.save_checkpoint = atomic_save
    started = time.monotonic()
    durations = []
    report_path = Path('/kaggle/working/train-report.json')
    def report(status):
        result = {'status':status,'target_epochs':trainer.num_epochs,'completed_epochs':trainer.current_epoch,
                  'fold':args.fold,'configuration':'3d_lowres','budget_hours':args.hours,
                  'elapsed_hours':(time.monotonic()-started)/3600,
                  'gpu':torch.cuda.get_device_name(0),'visible_gpu_count':torch.cuda.device_count(),
                  'trainer':trainer.__class__.__name__, 'full_volume_validation_performed':False,
                  'metric_note':'Training log pseudo-Dice is not full-volume validation Dice.',
                  **estimate(durations,trainer.current_epoch,trainer.num_epochs)}
        report_path.write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
    original_start, original_end = trainer.on_epoch_start, trainer.on_epoch_end
    epoch_started = None
    def start_epoch():
        nonlocal epoch_started
        elapsed = time.monotonic()-started
        predicted = statistics.median(durations[-5:]) if durations else 0
        if elapsed + predicted + 60 >= args.hours*3600:
            raise BudgetReached()
        epoch_started = time.monotonic()
        original_start()
    def end_epoch():
        original_end()  # increments current_epoch
        durations.append(time.monotonic()-epoch_started)
        report('training')
    trainer.on_epoch_start, trainer.on_epoch_end = start_epoch, end_epoch
    try:
        trainer.run_training()
        report('completed_100_epochs')
    except BudgetReached:
        # save_checkpoint stores current_epoch+1; here current_epoch already
        # points at the next epoch after on_epoch_end, so adjust temporarily.
        if trainer.current_epoch > 0:
            save_for_next_epoch(trainer, output/'checkpoint_latest.pth')
        report('paused_budget_checkpoint_saved' if trainer.current_epoch else 'paused_before_first_epoch')
    finally:
        for name in ('dataloader_train','dataloader_val'):
            loader = getattr(trainer,name,None)
            if hasattr(loader,'_finish'): loader._finish()
    # Deliberately skip perform_actual_validation here: it is a separate GPU
    # workload, to be budgeted explicitly after reviewing this pilot.


if __name__ == '__main__': main()
