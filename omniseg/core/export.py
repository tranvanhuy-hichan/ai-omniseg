"""Saving masks, meshes and reports. Mask/report code needs only NumPy and nibabel."""
import csv
import json
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np

DISCLAIMER = 'Kết quả nghiên cứu, chưa được kiểm định lâm sàng; không dùng để chẩn đoán.'
MESH_FORMATS = {'stl': 'STL', 'ply': 'PLY', 'obj': 'OBJ'}


def strip_nifti(name):
    return name.removesuffix('.nii.gz').removesuffix('.nii')


def save_mask(volume, mask, path):
    """Write `mask` (canonical RAS grid) as NIfTI in the same axes and affine as the opened image."""
    path = Path(path)
    if not path.name.endswith(('.nii', '.nii.gz')):
        path = path.with_name(path.name + '.nii.gz')
    data = volume.to_original(np.asarray(mask, dtype=np.uint8))
    affine = volume.orig_affine if volume.orig_affine is not None else volume.image.affine
    image = nib.Nifti1Image(data.astype(np.uint8), affine)
    image.set_data_dtype(np.uint8)
    nib.save(image, path)
    return path


def _writer(fmt):
    if fmt == 'stl':
        from vtkmodules.vtkIOGeometry import vtkSTLWriter
        writer = vtkSTLWriter(); writer.SetFileTypeToBinary()
    elif fmt == 'ply':
        from vtkmodules.vtkIOPLY import vtkPLYWriter
        writer = vtkPLYWriter(); writer.SetFileTypeToBinary()
    elif fmt == 'obj':
        from vtkmodules.vtkIOGeometry import vtkOBJWriter
        writer = vtkOBJWriter()
    else:
        raise ValueError(f'Định dạng mesh không hỗ trợ: {fmt}')
    return writer


def write_mesh(polydata, path, fmt=None):
    """Write one vtkPolyData (world mm, RAS) to STL/PLY/OBJ; the format defaults to the extension."""
    path = Path(path)
    fmt = (fmt or path.suffix.lstrip('.')).lower()
    writer = _writer(fmt)
    writer.SetFileName(str(path)); writer.SetInputData(polydata)
    if writer.Write() != 1 or not path.is_file():
        raise OSError(f'Không ghi được {path}')
    return path


def export_meshes(meshes, names, folder, base, fmt='stl', per_label=True, combined=True):
    """Write {label: polydata} as <base>_<name>.<fmt> per label and/or one <base>_all.<fmt>."""
    from vtkmodules.vtkFiltersCore import vtkAppendPolyData
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    if per_label:
        for label, mesh in meshes.items():
            safe = ''.join(c if c.isalnum() else '_' for c in names.get(label, f'label{label}')).strip('_') or f'label{label}'
            written.append(write_mesh(mesh, folder / f'{base}_{label}_{safe}.{fmt}', fmt))
    if combined and meshes:
        merge = vtkAppendPolyData()
        for mesh in meshes.values():
            merge.AddInputData(mesh)
        merge.Update()
        written.append(write_mesh(merge.GetOutput(), folder / f'{base}_all.{fmt}', fmt))
    if not written:
        raise ValueError('Không có bề mặt nào để xuất (mask trống).')
    return written


def build_report(volume, task, mask, source, entries, analysis=None, edited=False):
    """Plain-dict report of volumes (mL), lesion measurements and, if present, Dice/IoU."""
    stats = {row['label']: row for row in (analysis or {}).get('stats', [])}
    comparison = {row['label']: row for row in ((analysis or {}).get('comparison') or [])}
    regions = []
    for label, name, _ in entries:
        voxels = int(np.count_nonzero(mask == label))
        row = {'label': label, 'name': name, 'voxels': voxels,
               'volume_ml': None if volume.voxel_ml is None else voxels * volume.voxel_ml}
        if label in stats:
            s = stats[label]
            row.update(components=s['components'], max_diameter_mm=s['max_diameter_mm'],
                       mean_intensity=s['mean'], std_intensity=s['std'])
        if label in comparison:
            c = comparison[label]
            row.update(dice=c['dice'], iou=c['iou'], reference_volume_ml=c['reference_ml'], volume_difference_ml=c['delta_ml'])
        regions.append(row)
    return {
        'generated_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'case': strip_nifti(volume.path.name),
        'image_file': str(volume.path),
        'model': (task or {}).get('display_name') or (task or {}).get('id'),
        'task_id': (task or {}).get('id'),
        'mask_source': source,
        'mask_edited_manually': bool(edited),
        'voxel_spacing': [float(x) for x in volume.spacing],
        'image_shape': [int(x) for x in volume.data.shape[:3]],
        'regions': regions,
        'disclaimer': DISCLAIMER,
    }


def write_report(report, path):
    """Write the report as .json, or as one-row-per-region .csv (anything else is rejected)."""
    path = Path(path)
    if path.suffix.lower() == '.json':
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    elif path.suffix.lower() == '.csv':
        columns = ['case', 'model', 'mask_source', 'generated_at']
        region_columns = []
        for region in report['regions']:
            region_columns += [k for k in region if k not in region_columns]
        with path.open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.writer(handle)
            writer.writerow(columns + region_columns)
            for region in report['regions']:
                writer.writerow([report[c] for c in columns] + [('' if region.get(k) is None else region.get(k)) for k in region_columns])
    else:
        raise ValueError('Chỉ hỗ trợ .json hoặc .csv')
    return path
