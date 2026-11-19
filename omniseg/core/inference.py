"""Builds the nnUNetv2_predict environment/command for a given task.

Kept separate from the Qt worker so the command-building logic (which task.json
fields map to which nnU-Net CLI flags) can be tested without PySide6/VTK.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def nnunet_env(root=REPO_ROOT):
    """Default nnUNet_raw/preprocessed/results paths, independent of task."""
    return {
        'nnUNet_raw': str(root / 'data/nnUNet_raw'),
        'nnUNet_preprocessed': str(root / 'data/nnUNet_preprocessed'),
        'nnUNet_results': str(root / 'artifacts/nnUNet_results'),
    }


def build_predict_command(task, input_dir, output_dir, save_probabilities=False):
    """Returns the nnUNetv2_predict argv for running `task` on input_dir.

    save_probabilities also writes <case>.npz with the softmax output (for the uncertainty map).
    """
    command = [
        'nnUNetv2_predict',
        '-d', str(task['nnunet_dataset_id']),
        '-c', task['nnunet_config'],
        '-f', str(task['fold']),
        '-i', str(input_dir),
        '-o', str(output_dir),
        '-tr', task['trainer'],
        '-chk', task['checkpoint'],
    ]
    if save_probabilities:
        command.append('--save_probabilities')
    return command


def build_predict_env(task, root=REPO_ROOT):
    """Process environment for nnUNetv2_predict: repo defaults, caller's env wins."""
    env = dict(os.environ)
    for key, value in nnunet_env(root).items():
        env.setdefault(key, value)
    return env


def checkpoint_path(task, root=REPO_ROOT):
    """Where nnUNetv2_predict looks for task's checkpoint (caller's nnUNet_results wins)."""
    results = Path(os.environ.get('nnUNet_results') or nnunet_env(root)['nnUNet_results'])
    run = f"{task['trainer']}__nnUNetPlans__{task['nnunet_config']}"
    return results / task['nnunet_dataset_name'] / run / f"fold_{task['fold']}" / task['checkpoint']


def checkpoint_exists(task, root=REPO_ROOT):
    return checkpoint_path(task, root).is_file()


def channel_count(task):
    return len(task['channel_names'])
