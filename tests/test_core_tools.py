"""Masks, metrics, editing, reports, DICOM geometry and uncertainty; no Qt or VTK needed."""
import json
import nibabel as nib
import numpy as np
import pytest

from omniseg.core import editing, export, metrics, uncertainty
from omniseg.core.volume import Volume


def make_volume(tmp_path, shape=(6, 7, 8), affine=None, name='case.nii.gz'):
    affine = np.diag([2., 1., 3., 1.]) if affine is None else affine
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    image = nib.Nifti1Image(data, affine)
    image.header.set_xyzt_units('mm')
    nib.save(image, tmp_path / name)
    return Volume.load(tmp_path / name)


def test_dice_iou_and_empty_handling():
    ref = np.zeros((4, 4, 4), np.uint8); ref[:2] = 1
    pred = np.zeros_like(ref); pred[1:3] = 1
    rows = metrics.compare_masks(pred, ref, [1, 2], voxel_ml=0.5)
    assert rows[0]['dice'] == pytest.approx(0.5) and rows[0]['iou'] == pytest.approx(1 / 3)
    assert rows[0]['delta_ml'] == 0
    assert rows[1]['dice'] is None and rows[1]['iou'] is None          # absent from both
    pred[:] = 0
    assert metrics.compare_masks(pred, ref, [1])[0]['dice'] == 0.0     # missed entirely
    with pytest.raises(ValueError):
        metrics.compare_masks(pred, ref[:2], [1])


def test_region_stats_diameter_and_components():
    pytest.importorskip('scipy')
    mask = np.zeros((20, 20, 20), np.uint8); mask[2:5, 2:4, 2:4] = 1; mask[10:12, 10:12, 10:12] = 1
    image = np.full(mask.shape, 5., np.float32)
    s = metrics.region_stats(mask, 1, image, (2., 1., 1.), voxel_ml=0.002)
    assert s['components'] == 2 and s['voxels'] == 12 + 8
    assert s['mean'] == 5 and s['ml'] == pytest.approx(0.04)
    assert s['max_diameter_mm'] == pytest.approx(np.sqrt(4 ** 2 + 1 + 1))   # largest component only
    assert metrics.region_stats(mask, 2, image, (1, 1, 1))['max_diameter_mm'] is None


def test_brush_respects_spacing_and_eraser_keeps_other_labels():
    mask = np.zeros((11, 11, 3), np.uint8)
    assert editing.apply_brush(mask, 2, 1, (5, 5), 2.0, (1., 2., 1.), 1)
    plane = mask[:, :, 1]
    assert plane[5, 5] and plane[7, 5] and plane[5, 6] and not plane[5, 7]   # 2 mm = 2 voxels in x, 1 in y
    assert mask[:, :, 0].sum() == 0
    mask[5, 5, 1] = 2
    editing.apply_brush(mask, 2, 1, (5, 5), 5.0, (1., 1., 1.), 0, only=1)
    assert mask[5, 5, 1] == 2 and (mask == 1).sum() == 0
    assert not editing.apply_brush(mask, 2, 1, (99, 99), 1.0, (1, 1, 1), 1)


def test_history_undo_redo_round_trip():
    mask = np.zeros((6, 6, 6), np.uint8); history = editing.MaskHistory()
    history.begin()
    history.edit(mask, 2, 3, lambda: editing.apply_brush(mask, 2, 3, (2, 2), 1.5, (1, 1, 1), 1))
    history.edit(mask, 0, 2, lambda: editing.apply_brush(mask, 0, 2, (2, 3), 1.0, (1, 1, 1), 2))   # crosses the first slice
    assert history.commit(mask)
    painted = mask.copy()
    assert history.undo(mask) and not mask.any() and not history.undo(mask)
    assert history.redo(mask) and np.array_equal(mask, painted) and not history.redo(mask)
    history.begin(); history.edit(mask, 2, 0, lambda: None)
    assert not history.commit(mask)                                    # nothing changed: no entry


def test_save_mask_restores_original_orientation(tmp_path):
    flip = np.array([[-2., 0, 0, 10], [0, 1, 0, 0], [0, 0, 3, 0], [0, 0, 0, 1]])   # x axis points left
    volume = make_volume(tmp_path, affine=flip)
    assert not np.allclose(volume.image.affine, volume.orig_affine)
    mask = np.zeros(volume.data.shape[:3], np.uint8); mask[0, 1, 2] = 3
    out = export.save_mask(volume, mask, tmp_path / 'pred')
    assert out.name == 'pred.nii.gz'
    saved = nib.load(out)
    assert np.allclose(saved.affine, volume.orig_affine) and saved.shape == volume.data.shape[:3]
    assert saved.get_fdata().astype(np.uint8)[-1, 1, 2] == 3          # canonical x=0 is file x=last
    assert np.array_equal(volume.load_mask(out), mask)                # round-trips through load_mask


def test_report_json_and_csv(tmp_path):
    volume = make_volume(tmp_path)
    mask = np.zeros(volume.data.shape[:3], np.uint8); mask[:2, :2, :2] = 1
    analysis = metrics.analyse(volume, 0, mask, mask.copy(), [1, 2])
    entries = [(1, 'Gan', (0, 0, 0)), (2, 'U gan', (0, 0, 0))]
    report = export.build_report(volume, {'id': 't', 'display_name': 'Liver'}, mask, 'Kết quả AI', entries, analysis, edited=True)
    assert report['regions'][0]['volume_ml'] == pytest.approx(8 * 6e-3) and report['regions'][0]['dice'] == 1.0
    assert report['regions'][1]['dice'] is None and report['mask_edited_manually']
    export.write_report(report, tmp_path / 'r.json')
    assert json.loads((tmp_path / 'r.json').read_text(encoding='utf-8'))['case'] == 'case'
    export.write_report(report, tmp_path / 'r.csv')
    rows = (tmp_path / 'r.csv').read_text(encoding='utf-8-sig').splitlines()
    assert len(rows) == 3 and 'volume_ml' in rows[0]
    with pytest.raises(ValueError):
        export.write_report(report, tmp_path / 'r.txt')


def test_command_adds_probabilities_flag_only_when_asked():
    from omniseg.core import inference
    task = {'nnunet_dataset_id': 1, 'nnunet_config': '2d', 'fold': 0, 'trainer': 't', 'checkpoint': 'c'}
    assert '--save_probabilities' not in inference.build_predict_command(task, 'i', 'o')
    assert inference.build_predict_command(task, 'i', 'o', save_probabilities=True)[-1] == '--save_probabilities'


def test_entropy_is_zero_when_certain_and_one_when_uniform():
    certain = np.zeros((3, 2, 2, 2), np.float32); certain[0] = 1
    assert uncertainty.entropy_map(certain).max() == pytest.approx(0, abs=1e-6)
    uniform = np.full((3, 2, 2, 2), 1 / 3, np.float32)
    assert uncertainty.entropy_map(uniform).min() == pytest.approx(1, abs=1e-5)


def test_load_entropy_handles_nnunet_axis_order(tmp_path):
    flip = np.array([[-2., 0, 0, 10], [0, 1, 0, 0], [0, 0, 3, 0], [0, 0, 0, 1]])
    volume = make_volume(tmp_path, affine=flip)
    file_shape = volume.to_original(volume.channels[0]).shape
    probs = np.full((2,) + file_shape[::-1], .5, np.float32)           # (classes, z, y, x)
    probs[:, 0, 0, 0] = [1, 0]                                         # file voxel (0,0,0) is certain
    np.savez_compressed(tmp_path / 'p.npz', probabilities=probs)
    entropy = uncertainty.load_entropy(tmp_path / 'p.npz', volume)
    assert entropy.shape == volume.channels[0].shape
    assert entropy[-1, 0, 0] == pytest.approx(0, abs=1e-6) and entropy[0, 0, 0] == pytest.approx(1)


def test_dicom_affine_matches_what_simpleitk_writes_to_nifti(tmp_path):
    sitk = pytest.importorskip('SimpleITK')
    from omniseg.core.dicom import affine_from_geometry
    image = sitk.Image(5, 6, 7, sitk.sitkFloat32)
    image.SetOrigin((10., -20., 30.)); image.SetSpacing((0.7, 0.8, 2.5))
    image.SetDirection((0, 1, 0, -1, 0, 0, 0, 0, 1))
    sitk.WriteImage(image, str(tmp_path / 'x.nii.gz'))
    mine = affine_from_geometry(image.GetOrigin(), image.GetSpacing(), image.GetDirection())
    assert np.allclose(nib.load(tmp_path / 'x.nii.gz').affine, mine, atol=1e-5)


def test_dicom_series_round_trip(tmp_path):
    sitk = pytest.importorskip('SimpleITK')
    from omniseg.core import dicom
    series = tmp_path / 'series'; series.mkdir()
    writer = sitk.ImageFileWriter(); writer.KeepOriginalImageUIDOn()
    rng = np.random.default_rng(0)
    values = rng.integers(-500, 500, size=(4, 6, 5)).astype(np.int16)   # (z, y, x)
    for z in range(4):
        slice_ = sitk.GetImageFromArray(values[z][None])
        slice_.SetSpacing((0.8, 0.9, 2.0)); slice_.SetOrigin((0., 0., 2. * z))
        for key, val in {'0020|000e': '1.2.3.4', '0020|000d': '1.2.3', '0008|0060': 'CT', '0020|0013': str(z + 1),
                         '0020|0032': f'0\\0\\{2. * z}', '0020|0037': '1\\0\\0\\0\\1\\0', '0028|1052': '0', '0028|1053': '1',
                         '0010|0020': 'x'}.items():
            slice_.SetMetaData(key, val)
        writer.SetFileName(str(series / f'{z}.dcm')); writer.Execute(slice_)
    path = dicom.convert_series(series, cache_dir=tmp_path / 'cache')
    assert dicom.convert_series(series, cache_dir=tmp_path / 'cache') == path        # cached
    volume = Volume.load(path)
    assert volume.data.shape[:3] == (5, 6, 4)
    assert np.allclose(volume.spacing, [0.8, 0.9, 2.0], atol=1e-4)
    assert np.isin(volume.channels[0], values).all()
    with pytest.raises(ValueError):
        dicom.convert_series(tmp_path / 'cache', cache_dir=tmp_path / 'c2')
