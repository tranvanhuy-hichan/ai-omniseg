"""DICOM series import: converts a series folder to a cached NIfTI (needs SimpleITK)."""
import hashlib
from pathlib import Path

import numpy as np

CACHE_DIR = Path.home() / '.cache' / 'omniseg' / 'dicom'


def affine_from_geometry(origin, spacing, direction):
    """RAS affine (NIfTI convention) from SimpleITK's LPS origin / spacing / direction cosines."""
    columns = np.asarray(direction, dtype=float).reshape(3, 3) * np.asarray(spacing, dtype=float)
    lps = np.eye(4)
    lps[:3, :3] = columns
    lps[:3, 3] = origin
    return np.diag([-1., -1., 1., 1.]) @ lps


def list_series(folder):
    """[(series_id, [file names])] for the DICOM series found directly in `folder`, largest first."""
    try:
        import SimpleITK as sitk
    except ImportError as error:
        raise RuntimeError('Cần cài SimpleITK để mở DICOM: pip install SimpleITK') from error
    ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folder)) or []
    series = [(sid, list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folder), sid))) for sid in ids]
    return sorted(series, key=lambda item: -len(item[1]))


def convert_series(folder, cache_dir=CACHE_DIR):
    """Read the largest DICOM series in `folder` and return the path of its NIfTI conversion.

    Stored values are rescaled (Hounsfield units for CT). The conversion is cached by series.
    """
    import nibabel as nib
    import SimpleITK as sitk
    series = list_series(folder)
    if not series:
        raise ValueError('Không tìm thấy chuỗi DICOM nào trong thư mục này.')
    series_id, files = series[0]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1((series_id + str(len(files)) + str(Path(files[0]).stat().st_mtime_ns)).encode()).hexdigest()[:16]
    target = cache_dir / f'{key}.nii.gz'
    if target.is_file():
        return target
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    image = reader.Execute()
    array = sitk.GetArrayFromImage(image)          # (z, y, x) or (z, y, x, c)
    if array.ndim != 3:
        raise ValueError('Chỉ hỗ trợ chuỗi DICOM thang xám 3D.')
    data = np.ascontiguousarray(array.transpose(2, 1, 0)).astype(np.float32)
    affine = affine_from_geometry(image.GetOrigin(), image.GetSpacing(), image.GetDirection())
    nifti = nib.Nifti1Image(data, affine)
    nifti.header.set_xyzt_units('mm')
    nib.save(nifti, target)
    return target
