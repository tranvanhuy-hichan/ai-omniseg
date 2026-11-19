import os

import numpy as np
import pytest

from omniseg import registry
from omniseg.core.anatomy import (CT_WINDOW, PROFILES, anatomy_kind, opacity_points, smooth, to_scalars)


def hu_to_scalar(hu):
    low, high = CT_WINDOW
    return int(np.clip((hu - low) / (high - low) * 255, 0, 255))


def opacity_at(kind, strength, scalar):
    points = opacity_points(kind, strength)
    return float(np.interp(scalar, [p[0] for p in points], [p[1] for p in points]))


def test_modality_decides_kind_and_air_detects_ct_otherwise():
    mri = np.full((4, 4, 4), 300., dtype=np.float32)
    ct = mri.copy(); ct[0, 0, 0] = -1000.
    assert anatomy_kind('CT', mri) == 'CT'
    assert anatomy_kind('MRI', mri) == 'MRI'
    assert anatomy_kind(None, mri) == 'MRI'
    assert anatomy_kind('MRI', ct) == 'CT' and anatomy_kind(None, ct) == 'CT'


def test_ct_uses_a_fixed_hu_window_and_keeps_air_at_zero():
    raw = np.array([-1000., -200., 40., 1000., 3000.], dtype=np.float32).reshape(5, 1, 1)
    assert to_scalars(raw, 'CT').ravel().tolist() == [0, 0, 51, 255, 255]
    # The MRI path still uses the per-image window and keeps exact zeros transparent.
    mri = np.array([0., 50., 100.], dtype=np.float32).reshape(3, 1, 1)
    assert to_scalars(mri, 'MRI', (0., 100.)).ravel().tolist() == [0, 127, 255]


@pytest.mark.parametrize('strength', [.05, .12, .5, 1.])
def test_ct_soft_tissue_is_gated_far_below_bone_at_every_strength(strength):
    soft, bone = hu_to_scalar(40), hu_to_scalar(500)
    assert opacity_at('CT', strength, hu_to_scalar(-1000)) == 0
    assert opacity_at('CT', strength, hu_to_scalar(160)) == 0   # contrast-enhanced vessels stay clear
    assert opacity_at('CT', strength, bone) > opacity_at('CT', strength, soft)
    assert all(0 <= amount <= 1 for _, amount in opacity_points('CT', strength))


def test_ct_profile_hides_homogeneous_soft_tissue_with_a_gradient_ramp():
    gradient = PROFILES['CT']['gradient']
    assert gradient[0] == (0, 0) and gradient[1][1] == 0 and gradient[-1][1] == 1
    assert PROFILES['MRI']['gradient'] is None and PROFILES['MRI']['shade'] is False


def test_smooth_spreads_a_spike_without_changing_total_intensity():
    volume = np.zeros((9, 9, 9), dtype=np.uint8); volume[4, 4, 4] = 200
    blurred = smooth(volume, passes=1)
    assert blurred[4, 4, 4] < 200 and blurred[3, 4, 4] > 0
    assert abs(int(blurred.sum()) - 200) <= 3 and blurred.dtype == np.uint8


def test_viewer_creates_ct_anatomy_with_ct_profile(tmp_path):
    import nibabel as nib
    from omniseg.core.anatomy import create_anatomy, set_anatomy_opacity
    from omniseg.core.volume import Volume
    data = np.full((8, 8, 8), -1000., dtype=np.float32); data[2:6, 2:6, 2:6] = 40.; data[3:5, 3:5, 3:5] = 600.
    path = tmp_path / 'ct.nii.gz'; nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    actor = create_anatomy(Volume.load(path), 0, 'CT')
    assert actor.omniseg_kind == 'CT'
    set_anatomy_opacity(actor, .5)
    assert actor.GetProperty().GetScalarOpacity().GetValue(0) == 0


def test_label_opacities_come_from_task_json_and_default_to_opaque():
    liver = registry.get_task('liver_tumour_ct')
    assert registry.label_opacities(liver) == {1: pytest.approx(.45)}
    brain = registry.get_task('brain_tumour_mri')
    assert registry.label_opacities(brain)[1] < 1 and 2 not in registry.label_opacities(brain)
    assert registry.label_opacities({'labels': {'background': 0, 'a': 1}, 'label_display': {'1': {'opacity': 7}}}) == {1: 1.0}
    assert registry.label_opacities({'labels': {'background': 0, 'a': 1}}) == {}


def test_dragging_the_camera_keeps_depth_peeling_and_the_image_volume(tmp_path):
    """Turning peeling off (or hiding the volume) during a drag washes the mask out; guard against it."""
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    pytest.importorskip('PySide6')
    import nibabel as nib
    from PySide6.QtWidgets import QApplication
    from omniseg.core.volume import Volume
    from omniseg.view3d import SurfaceViewer
    QApplication.instance() or QApplication([])
    data = np.full((8, 8, 8), -1000., dtype=np.float32); data[2:6, 2:6, 2:6] = 600.
    path = tmp_path / 'ct.nii.gz'; nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    viewer = SurfaceViewer(registry.label_entries(registry.get_task('liver_tumour_ct')))
    viewer.set_anatomy(Volume.load(path), 0, 'CT')
    interactor = viewer.vtk.GetRenderWindow().GetInteractor()
    for event in ('StartInteractionEvent', 'EndInteractionEvent'):
        interactor.InvokeEvent(event)
        assert viewer.renderer.GetUseDepthPeeling() and viewer.renderer.GetUseDepthPeelingForVolumes()
        assert viewer.anatomy_actor.GetVisibility()
