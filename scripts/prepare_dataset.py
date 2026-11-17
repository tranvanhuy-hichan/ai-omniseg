"""Validate co-registered NIfTI cases and copy them into nnU-Net v2 format."""
import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path


def validate_case(case, channels, labels, base):
    import nibabel as nib
    import numpy as np

    if not re.fullmatch(r"[A-Za-z0-9-]+", case["id"]):
        raise ValueError("Case IDs must contain only letters, digits or hyphens")
    if len(case["images"]) != channels:
        raise ValueError(f"{case['id']}: expected {channels} image channels")
    paths = [(base / p).resolve() for p in [*case["images"], case["label"]]]
    if any(not p.name.endswith('.nii.gz') for p in paths):
        raise ValueError("This importer requires .nii.gz files")
    if len(set(paths)) != len(paths):
        raise ValueError("Each image channel and mask must be a distinct file")
    reference = None
    for path in paths:
        image = nib.load(path)
        values = np.asarray(image.dataobj)
        if values.ndim != 3 or not np.isfinite(values).all():
            raise ValueError(f"{path}: expected a finite 3D volume")
        if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-12:
            raise ValueError(f"{path}: invalid voxel-to-world affine")
        if reference is None:
            reference = image
        elif image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-4, rtol=0):
            raise ValueError(f"{case['id']}: image/mask geometry does not match; co-register first")
        if path == paths[-1] and not set(np.unique(values)).issubset(labels):
            raise ValueError(f"{path}: mask has labels outside {sorted(labels)}; convert labels explicitly first")
    return paths


def prepare(config_path, manifest_path, raw_root):
    config = json.loads(config_path.read_text())
    cases = json.loads(manifest_path.read_text())
    name = config["dataset_name"]
    if not re.fullmatch(r"Dataset\d{3}_[A-Za-z][A-Za-z0-9]*", name):
        raise ValueError("Expected dataset_name like Dataset501_BrainTumor")
    channels = config["channel_names"]
    if set(channels) != {str(i) for i in range(len(channels))} or not channels:
        raise ValueError("Channel keys must be consecutive strings starting at 0")
    labels = set(config["labels"].values())
    if (any(type(v) is not int for v in config['labels'].values())
            or config["labels"].get("background") != 0
            or labels != set(range(len(labels))) or len(labels) < 2
            or len(labels) != len(config['labels'])):
        raise ValueError("Use unique consecutive integer labels, background=0 and at least one foreground class")
    if config.get('file_ending') != '.nii.gz':
        raise ValueError('This importer supports only .nii.gz')
    if not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError("Manifest must contain cases with unique IDs")
    target = raw_root / name
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    # Validate the entire manifest before copying anything.
    validated = [validate_case(c, len(channels), labels, manifest_path.parent) for c in cases]
    if len({paths[-1] for paths in validated}) != len(cases):
        raise ValueError('Multiple cases reference the same mask; check for duplicate patients')
    raw_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.prepare-', dir=raw_root) as temp:
        staged = Path(temp) / name
        (staged / 'imagesTr').mkdir(parents=True)
        (staged / 'labelsTr').mkdir()
        for case, paths in zip(cases, validated):
            for channel, source in enumerate(paths[:-1]):
                shutil.copy2(source, staged / 'imagesTr' / f"{case['id']}_{channel:04d}.nii.gz")
            shutil.copy2(paths[-1], staged / 'labelsTr' / f"{case['id']}.nii.gz")
        metadata = {k: config[k] for k in ('channel_names', 'labels', 'file_ending')}
        metadata['numTraining'] = len(cases)
        (staged / 'dataset.json').write_text(json.dumps(metadata, indent=2) + '\n')
        staged.rename(target)
    return target


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--raw-root', type=Path, default=Path(os.environ.get('nnUNet_raw', root / 'data/nnUNet_raw')))
    args = parser.parse_args()
    print(prepare(args.config.resolve(), args.manifest.resolve(), args.raw_root.resolve()))


if __name__ == '__main__':
    main()
