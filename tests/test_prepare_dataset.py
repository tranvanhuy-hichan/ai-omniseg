import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    'prepare_dataset', Path(__file__).resolve().parents[1] / 'scripts/prepare_dataset.py'
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture_files(tmp_path, *, wrong_geometry=False, wrong_label=False):
    shape = (8, 9, 10)
    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    image = tmp_path / 'image.nii.gz'
    label = tmp_path / 'mask.nii.gz'
    nib.save(nib.Nifti1Image(np.ones(shape, dtype=np.float32), affine), image)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[2:4, 2:4, 2:4] = 4 if wrong_label else 1
    mask_affine = affine.copy()
    if wrong_geometry:
        mask_affine[0, 3] = 20
    nib.save(nib.Nifti1Image(mask, mask_affine), label)
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({
        'dataset_name': 'Dataset501_BrainTumor', 'channel_names': {'0': 'FLAIR'},
        'labels': {'background': 0, 'tumor': 1}, 'file_ending': '.nii.gz',
    }))
    manifest = tmp_path / 'cases.json'
    manifest.write_text(json.dumps([{'id': 'patient001', 'images': [image.name], 'label': label.name}]))
    return config, manifest


def test_import_preserves_geometry_and_refuses_overwrite(tmp_path):
    config, manifest = fixture_files(tmp_path)
    target = module.prepare(config, manifest, tmp_path / 'raw')
    metadata = json.loads((target / 'dataset.json').read_text())
    assert metadata['numTraining'] == 1
    image = nib.load(target / 'imagesTr/patient001_0000.nii.gz')
    mask = nib.load(target / 'labelsTr/patient001.nii.gz')
    assert np.array_equal(image.affine, mask.affine)
    assert image.header.get_zooms() == (1, 1, 2)
    with pytest.raises(FileExistsError):
        module.prepare(config, manifest, tmp_path / 'raw')


@pytest.mark.parametrize('kwargs, message', [
    ({'wrong_geometry': True}, 'geometry'), ({'wrong_label': True}, 'labels outside'),
])
def test_invalid_case_does_not_leave_partial_dataset(tmp_path, kwargs, message):
    config, manifest = fixture_files(tmp_path, **kwargs)
    with pytest.raises(ValueError, match=message):
        module.prepare(config, manifest, tmp_path / 'raw')
    assert not (tmp_path / 'raw/Dataset501_BrainTumor').exists()


def test_duplicate_patient_ids_rejected(tmp_path):
    config, manifest = fixture_files(tmp_path)
    cases = json.loads(manifest.read_text())
    manifest.write_text(json.dumps(cases * 2))
    with pytest.raises(ValueError, match='unique IDs'):
        module.prepare(config, manifest, tmp_path / 'raw')
