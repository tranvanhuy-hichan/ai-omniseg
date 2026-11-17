"""Verify command forwarding without loading nnU-Net or starting training."""
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('args, command, expected', [
    (['plan', '501', '2d'], 'nnUNetv2_plan_and_preprocess',
     ['-d', '501', '-c', '2d', '--verify_dataset_integrity']),
    (['train', '501', '2d', '0', '--npz'], 'nnUNetv2_train',
     ['501', '2d', '0', '--npz']),
    (['predict', '501', '2d', '0', 'input with spaces', 'output with spaces'], 'nnUNetv2_predict',
     ['-d', '501', '-c', '2d', '-f', '0', '-i', 'input with spaces', '-o', 'output with spaces']),
])
def test_exact_command_arguments(tmp_path, args, command, expected):
    binary = tmp_path / command
    binary.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
    binary.chmod(0o755)
    env = {**os.environ, 'PATH': f"{tmp_path}:{os.environ['PATH']}",
           'nnUNet_raw': str(tmp_path / 'raw'),
           'nnUNet_preprocessed': str(tmp_path / 'preprocessed'),
           'nnUNet_results': str(tmp_path / 'results')}
    completed = subprocess.run(['bash', str(ROOT / 'scripts/pipeline.sh'), *args],
                               env=env, capture_output=True, text=True, check=True)
    assert completed.stdout.splitlines() == expected
