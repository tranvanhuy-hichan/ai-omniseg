import nibabel as nib
import numpy as np
import pytest

from desktop.volume import Volume, plane, render_slice


def make_volume(tmp_path):
    path = tmp_path / 'mri.nii.gz'
    image = nib.Nifti1Image(np.arange(4*5*6*4, dtype=np.float32).reshape(4,5,6,4), np.diag([1,2,3,1]))
    image.header.set_xyzt_units('mm')
    nib.save(image, path)
    return Volume.load(path)


def test_physical_volume_and_mask_geometry(tmp_path):
    volume = make_volume(tmp_path)
    mask = np.zeros((4,5,6), dtype=np.uint8)
    mask[1,1,1] = 3
    assert volume.label_volumes(mask)[2] == pytest.approx(.006)
    path = tmp_path / 'mask.nii.gz'
    nib.save(nib.Nifti1Image(mask, np.diag([1,1,1,1])), path)
    with pytest.raises(ValueError, match='affine'):
        volume.load_mask(path)


def test_channel_slice_and_overlay(tmp_path):
    volume = make_volume(tmp_path)
    assert volume.names == ['Kênh 1', 'Kênh 2', 'Kênh 3', 'Kênh 4']
    for axis in range(3):
        image = render_slice(volume, axis, 1, 3, (0, 500))
        assert image.dtype == np.uint8 and image.shape[-1] == 3
        assert image.flags.c_contiguous
    mask = np.ones(volume.data.shape[:3], dtype=np.uint8)
    plain = render_slice(volume, 2, 1, 0, (0,500))
    overlay = render_slice(volume, 2, 1, 0, (0,500), mask, .5)
    assert not np.array_equal(plain, overlay)


def test_canonical_orientation_and_display(tmp_path):
    data = np.arange(24, dtype=np.float32).reshape(2,3,4)
    path = tmp_path / 'las.nii.gz'
    nib.save(nib.Nifti1Image(data, np.diag([-1,1,1,1])), path)
    volume = Volume.load(path)
    assert nib.aff2axcodes(volume.image.affine) == ('R','A','S')
    assert np.array_equal(volume.data[...,0], data[::-1])
    assert np.array_equal(plane(data, 2, 0), np.flipud(data[:,:,0].T))
