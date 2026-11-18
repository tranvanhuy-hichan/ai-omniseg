"""Verify the chosen MSD task's metadata and invoke the upstream 4D-to-3D converter."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from omniseg.registry import get_task

# Per-task fingerprint of the *raw* MSD archive's dataset.json, keyed by task
# id. This is deliberately separate from tasks/<id>/task.json: task.json
# describes the trained checkpoint's channel/label naming convention, while
# these are the exact (sometimes differently spelled) strings the official
# MSD archive itself ships, used only to verify we extracted the right thing.
MSD_SOURCE = {
    'brain_tumour_mri': {
        'folder': 'Task01_BrainTumour',
        'modality': {'0': 'FLAIR', '1': 'T1w', '2': 't1gd', '3': 'T2w'},
        'labels': {'0': 'background', '1': 'edema', '2': 'non-enhancing tumor', '3': 'enhancing tumour'},
    },
    'liver_tumour_ct': {
        'folder': 'Task03_Liver',
        'modality': {'0': 'CT'},
        'labels': {'0': 'background', '1': 'liver', '2': 'cancer'},
    },
}


def verify(source, task):
    expected = MSD_SOURCE.get(task['id'])
    if expected is None:
        raise ValueError(f"No known MSD source fingerprint for task '{task['id']}'")
    if source.name != expected['folder']:
        raise ValueError(f"Expected extracted {expected['folder']} directory")
    metadata = json.loads((source / 'dataset.json').read_text())
    if metadata.get('modality') != expected['modality'] or metadata.get('labels') != expected['labels']:
        raise ValueError('Source channels/labels differ from the selected MSD release; do not guess or remap silently')
    for name in ('imagesTr', 'labelsTr'):
        if not (source / name).is_dir():
            raise ValueError(f'Missing {name}')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', default='brain_tumour_mri',
                        help='Task id under tasks/ (default: brain_tumour_mri)')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--dataset-id', type=int, default=None,
                        help='Defaults to the selected task\'s nnunet_dataset_id')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    task = get_task(args.task)
    if task is None:
        parser.error(f"Unknown task '{args.task}'; see tasks/README.md")
    dataset_id = args.dataset_id if args.dataset_id is not None else task['nnunet_dataset_id']
    if not 1 <= dataset_id <= 999 or args.workers < 1:
        parser.error('Use dataset ID 1..999 and at least one worker')
    source = args.source.resolve()
    verify(source, task)
    env = dict(os.environ)
    for key, default in [('nnUNet_raw', ROOT / 'data/nnUNet_raw'),
                         ('nnUNet_preprocessed', ROOT / 'data/nnUNet_preprocessed'),
                         ('nnUNet_results', ROOT / 'artifacts/nnUNet_results')]:
        env.setdefault(key, str(default))
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    if list(Path(env['nnUNet_raw']).glob(f'Dataset{dataset_id:03d}_*')):
        raise FileExistsError('Dataset ID already exists; choose a new ID or inspect the existing dataset')
    subprocess.run(['nnUNetv2_convert_MSD_dataset', '-i', str(source),
                    '-overwrite_id', str(dataset_id), '-np', str(args.workers)],
                   env=env, check=True)


if __name__ == '__main__':
    main()
