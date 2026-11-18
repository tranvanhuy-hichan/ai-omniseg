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


def build_predict_command(task, input_dir, output_dir):
    """Returns the nnUNetv2_predict argv for running `task` on input_dir."""
    return [
        'nnUNetv2_predict',
        '-d', str(task['nnunet_dataset_id']),
        '-c', task['nnunet_config'],
        '-f', str(task['fold']),
        '-i', str(input_dir),
        '-o', str(output_dir),
        '-tr', task['trainer'],
        '-chk', task['checkpoint'],
    ]


def build_predict_env(task, root=REPO_ROOT):
    """Process environment for nnUNetv2_predict: repo defaults, caller's env wins."""
    env = dict(os.environ)
    for key, value in nnunet_env(root).items():
        env.setdefault(key, value)
    return env


def channel_count(task):
    return len(task['channel_names'])
