"""Verify the chosen MSD Task01 metadata and invoke the upstream 4D-to-3D converter."""
import argparse
import json
import os
import subprocess
from pathlib import Path


def verify(source):
    if source.name != 'Task01_BrainTumour':
        raise ValueError('Expected extracted Task01_BrainTumour directory')
    metadata = json.loads((source / 'dataset.json').read_text())
    expected_channels = {'0': 'FLAIR', '1': 'T1w', '2': 't1gd', '3': 'T2w'}
    expected_labels = {'0': 'background', '1': 'edema', '2': 'non-enhancing tumor', '3': 'enhancing tumour'}
    if metadata.get('modality') != expected_channels or metadata.get('labels') != expected_labels:
        raise ValueError('Source channels/labels differ from the selected MSD release; do not guess or remap silently')
    for name in ('imagesTr', 'labelsTr'):
        if not (source / name).is_dir():
            raise ValueError(f'Missing {name}')
    return metadata


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--dataset-id', type=int, default=501)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.dataset_id <= 999 or args.workers < 1:
        parser.error('Use dataset ID 1..999 and at least one worker')
    source = args.source.resolve()
    verify(source)
    env = dict(os.environ)
    for key, default in [('nnUNet_raw', root / 'data/nnUNet_raw'),
                         ('nnUNet_preprocessed', root / 'data/nnUNet_preprocessed'),
                         ('nnUNet_results', root / 'artifacts/nnUNet_results')]:
        env.setdefault(key, str(default))
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    if list(Path(env['nnUNet_raw']).glob(f'Dataset{args.dataset_id:03d}_*')):
        raise FileExistsError('Dataset ID already exists; choose a new ID or inspect the existing dataset')
    subprocess.run(['nnUNetv2_convert_MSD_dataset', '-i', str(source),
                    '-overwrite_id', str(args.dataset_id), '-np', str(args.workers)],
                   env=env, check=True)


if __name__ == '__main__':
    main()
