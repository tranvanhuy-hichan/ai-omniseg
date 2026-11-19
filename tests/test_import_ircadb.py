import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    'import_ircadb', Path(__file__).resolve().parents[1] / 'scripts/import_ircadb.py'
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_merge_labels_uses_task_convention_and_cancer_overrides_liver():
    liver = np.zeros((4, 4, 4), bool)
    liver[1:3] = True
    tumour_a = np.zeros_like(liver)
    tumour_a[1, 1, 1] = True
    tumour_b = np.zeros_like(liver)
    tumour_b[2, 2, 2] = True
    label, stats = module.merge_labels(liver, [tumour_a, tumour_b])
    assert label.dtype == np.uint8
    assert set(np.unique(label)) == {0, 1, 2}
    assert label[1, 1, 1] == 2 and label[2, 2, 2] == 2
    assert label[1, 0, 0] == 1 and label[0, 0, 0] == 0
    assert stats == {'liver_voxels': 32, 'tumour_voxels': 2, 'tumour_outside_liver_voxels': 0}


def test_merge_labels_counts_tumour_outside_liver_and_still_labels_it():
    liver = np.zeros((3, 3, 3), bool)
    liver[0] = True
    tumour = np.zeros_like(liver)
    tumour[2, 0, 0] = True
    label, stats = module.merge_labels(liver, [tumour])
    assert stats['tumour_outside_liver_voxels'] == 1
    assert label[2, 0, 0] == 2


def test_merge_labels_without_tumour_regions_gives_liver_only():
    liver = np.ones((2, 2, 2), bool)
    label, stats = module.merge_labels(liver, [])
    assert set(np.unique(label)) == {1} and stats['tumour_voxels'] == 0


def synthetic_ct():
    array = np.full((6, 8, 8), -500.0)
    array[:3, 2:6, 2:6] = 110.0
    liver = np.zeros(array.shape, bool)
    liver[:3, 2:6, 2:6] = True
    return array, liver


def test_choose_orientation_accepts_aligned_mask():
    array, liver = synthetic_ct()
    flipped, mean_hu = module.choose_orientation(array, liver)
    assert flipped is False and mean_hu == 110.0


def test_choose_orientation_detects_reversed_slice_order():
    array, liver = synthetic_ct()
    flipped, mean_hu = module.choose_orientation(array, liver[::-1])
    assert flipped is True and mean_hu == 110.0


def test_choose_orientation_refuses_when_neither_order_looks_like_liver():
    array = np.full((6, 8, 8), -500.0)
    liver = np.zeros(array.shape, bool)
    liver[:3, 2:6, 2:6] = True
    assert module.choose_orientation(array, liver) == (False, None)
