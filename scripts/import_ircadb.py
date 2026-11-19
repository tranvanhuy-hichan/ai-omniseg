"""Download 3D-IRCADb-01 and convert it to NIfTI as an external test set for liver_tumour_ct.

Labels follow tasks/liver_tumour_ct/task.json: 0 background, 1 liver, 2 cancer.
The dataset is independent of MSD Task03_Liver / LiTS. License: CC BY-NC-ND 4.0.
"""
import argparse
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np

URL = 'https://cloud.ircad.fr/index.php/s/JN3z7EynBiwYyjy/download?path=%2F3Dircadb1.{n}'
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'data/test_sets/liver_tumour_ct'
LIVER_DIR = 'liver'
TUMOUR_PREFIX = 'livertumor'


def merge_labels(liver, tumours):
    """Return (label, stats) from boolean liver mask and a list of boolean tumour masks."""
    label = np.zeros(liver.shape, np.uint8)
    label[liver] = 1
    tumour = np.zeros(liver.shape, bool)
    for mask in tumours:
        tumour |= mask
    outside = int((tumour & ~liver).sum())
    label[tumour] = 2
    return label, {'liver_voxels': int(liver.sum()), 'tumour_voxels': int(tumour.sum()),
                   'tumour_outside_liver_voxels': outside}


PLAUSIBLE_LIVER_HU = (20, 200)


def choose_orientation(array, liver):
    """Check the mask is aligned with the CT by mean HU inside the liver mask.

    Returns (flipped, mean_hu); mean_hu is None if neither slice order gives a plausible liver.
    """
    low, high = PLAUSIBLE_LIVER_HU
    as_is = float(array[liver].mean())
    if low <= as_is <= high:
        return False, as_is
    flipped = float(array[liver[::-1]].mean())
    if low <= flipped <= high:
        return True, flipped
    return False, None


def read_series(folder):
    """Read a DICOM folder as one 3D image, slices ordered by their patient position."""
    import SimpleITK as sitk  # imported lazily so the pure-numpy helpers stay testable without it
    names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folder))
    if not names:
        raise ValueError(f'No DICOM series in {folder}')
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(names)
    return reader.Execute()


def read_mask(folder):
    import SimpleITK as sitk
    return sitk.GetArrayFromImage(read_series(folder)) > 0


def download(n, target):
    subprocess.run(['curl', '-fL', '--retry', '5', '-C', '-', '-sS', '-o', str(target), URL.format(n=n)], check=True)


def extract_case(archive, work):
    with zipfile.ZipFile(archive) as outer:
        outer.extractall(work)
    case = next(p for p in work.iterdir() if p.is_dir())
    for name in ('PATIENT_DICOM', 'MASKS_DICOM'):
        with zipfile.ZipFile(case / f'{name}.zip') as inner:
            inner.extractall(case)
    return case


def convert_case(n, out):
    import SimpleITK as sitk
    name = f'ircadb_{n:02d}'
    image_path, label_path = out / 'images' / f'{name}.nii.gz', out / 'labels' / f'{name}.nii.gz'
    if image_path.is_file() and label_path.is_file():
        return {'case': name, 'status': 'skipped_existing'}
    with tempfile.TemporaryDirectory(prefix=f'ircadb{n}-') as tmp:
        tmp = Path(tmp)
        download(n, tmp / 'case.zip')
        case = extract_case(tmp / 'case.zip', tmp / 'x')
        image = read_series(case / 'PATIENT_DICOM')
        masks = case / 'MASKS_DICOM'
        regions = sorted(p.name for p in masks.iterdir() if p.is_dir())
        if LIVER_DIR not in regions:
            return {'case': name, 'status': 'failed', 'reason': f'no liver mask, regions={regions}'}
        tumour_dirs = [r for r in regions if r.startswith(TUMOUR_PREFIX)]
        liver_img = read_series(masks / LIVER_DIR)
        if liver_img.GetSize() != image.GetSize():
            return {'case': name, 'status': 'failed',
                    'reason': f'size mismatch image={image.GetSize()} liver={liver_img.GetSize()}'}
        liver = sitk.GetArrayFromImage(liver_img) > 0
        tumours = [read_mask(masks / d) for d in tumour_dirs]
        if any(t.shape != liver.shape for t in tumours):
            return {'case': name, 'status': 'failed', 'reason': 'tumour mask shape mismatch'}
        array = sitk.GetArrayFromImage(image)
        flipped, mean_hu = choose_orientation(array, liver)
        if mean_hu is None:
            return {'case': name, 'status': 'failed',
                    'reason': 'liver mask does not sit on plausible liver HU in either slice order; not guessing'}
        if flipped:
            liver, tumours = liver[::-1], [t[::-1] for t in tumours]
        label, stats = merge_labels(liver, tumours)
        label_img = sitk.GetImageFromArray(label)
        label_img.CopyInformation(image)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(sitk.Cast(image, sitk.sitkInt16), str(image_path))
        sitk.WriteImage(label_img, str(label_path))
        return {'case': name, 'status': 'converted', 'size_xyz': image.GetSize(),
                'spacing_xyz': tuple(round(s, 3) for s in image.GetSpacing()),
                'liver_mean_hu': round(mean_hu, 1), 'z_flipped': flipped, 'tumour_regions': tumour_dirs,
                'ignored_regions': [r for r in regions if r != LIVER_DIR and r not in tumour_dirs],
                **stats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--cases', type=int, nargs='+', default=list(range(1, 21)))
    args = parser.parse_args()
    if any(not 1 <= n <= 20 for n in args.cases):
        parser.error('Cases must be in 1..20')
    for n in args.cases:
        print(convert_case(n, args.out), flush=True)


if __name__ == '__main__':
    main()
