"""Optional headless Qt tests; lightweight CI can skip without PySide6."""
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import nibabel as nib
import numpy as np
import pytest

pytest.importorskip('PySide6')
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from omniseg.app import STYLE, Workspace
from omniseg.core.volume import Volume


@pytest.fixture
def window(tmp_path):
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(STYLE)
    viewer = Workspace()
    path = tmp_path / 'volume.nii.gz'
    nib.save(nib.Nifti1Image(np.ones((10, 12, 14), dtype=np.float32), np.eye(4)), path)
    viewer.set_volume(Volume.load(path))
    viewer.show()
    app.processEvents()
    yield viewer, app
    viewer.close()
    app.processEvents()


def test_expansion_restores_layout_and_slice(window):
    viewer, app = window
    panel = viewer.panels[1]
    panel.slider.setValue(3)
    before = panel.canvas.width()
    QTest.mouseClick(panel.expand_button, Qt.MouseButton.LeftButton)
    app.processEvents()
    assert viewer.focus_axis == 1
    assert panel.canvas.width() > before
    assert viewer.viewer_stack.currentWidget() is viewer.focus_viewer
    assert panel.parentWidget() is viewer.focus_viewer
    assert sum(p.isVisible() for p in viewer.panels) == 1
    panel.slider.setValue(5)
    QTest.keyClick(viewer, Qt.Key.Key_Escape)
    app.processEvents()
    assert viewer.focus_axis is None
    assert viewer.viewer_stack.currentWidget() is viewer.viewer
    assert all(p.parentWidget() is viewer.viewer and p.isVisible() for p in viewer.panels)
    assert viewer.library.isVisible() and viewer.inspector.isVisible()
    assert panel.slider.value() == 5


def test_double_click_and_layout_selector(window):
    viewer, app = window
    panel = viewer.panels[0]
    QTest.mouseDClick(panel.canvas, Qt.MouseButton.LeftButton)
    app.processEvents()
    assert viewer.focus_axis == panel.axis
    QTest.mouseDClick(panel.canvas, Qt.MouseButton.LeftButton)
    app.processEvents()
    assert viewer.focus_axis is None
    viewer.layout_choice.setCurrentIndex(viewer.layout_choice.findData(0))
    app.processEvents()
    assert viewer.focus_axis == 0
    assert viewer.panels[2].isVisible()


def test_every_plane_replaces_the_entire_image_area(window):
    viewer, app = window
    for axis in (2, 1, 0, 2):
        viewer.set_focus(axis)
        app.processEvents()
        selected = next(p for p in viewer.panels if p.axis == axis)
        assert sum(p.isVisible() for p in viewer.panels) == 1
        assert selected.width() >= viewer.viewer_stack.width() - 2
        assert selected.height() >= viewer.viewer_stack.height() - 2
    viewer.set_focus(None)
    app.processEvents()
    assert all(p.isVisible() for p in viewer.panels)
    # The complete app header plus case toolbar should leave space for images.
    assert viewer.viewer_stack.mapTo(viewer, viewer.viewer_stack.rect().topLeft()).y() <= 110


def region_names(viewer):
    from PySide6.QtWidgets import QLabel
    names = []
    for i in range(viewer.region_rows.count()):
        texts = [l.text() for l in viewer.region_rows.itemAt(i).widget().findChildren(QLabel)]
        names.append(texts[1])  # [dot, name, amount]
    return names


def pick_model(viewer, task_id):
    viewer.model_choice.setCurrentIndex(next(i for i in range(viewer.model_choice.count())
                                             if viewer.model_choice.itemData(i)['id'] == task_id))


def test_region_panel_follows_selected_task(window):
    viewer, app = window
    pick_model(viewer, 'brain_tumour_mri')
    assert region_names(viewer) == ['Phù nề', 'U không tăng tương phản', 'U tăng tương phản']
    assert viewer.label_ids == [1, 2, 3] and 'BRAIN TUMOUR' in viewer.region_title.text()
    pick_model(viewer, 'liver_tumour_ct')
    assert region_names(viewer) == ['Gan', 'U gan']
    assert viewer.label_ids == [1, 2] and 'LIVER' in viewer.region_title.text()
    assert 'MSD TASK01' not in viewer.region_title.text()


def test_liver_mask_fills_only_its_rows_and_model_switch_drops_it(window, tmp_path):
    viewer, app = window
    pick_model(viewer, 'liver_tumour_ct')
    mask = np.zeros((10, 12, 14), dtype=np.uint8); mask[2:5, 2:5, 2:5] = 1; mask[3, 3, 3] = 2
    viewer.set_mask(mask, tmp_path / 'mask.nii.gz')
    assert len(viewer.region_labels) == 2
    assert '26 voxel' in viewer.region_labels[0].text()
    assert '1 voxel' in viewer.region_labels[1].text()
    assert all(t.isEnabled() for t in viewer.region_toggles)
    pick_model(viewer, 'brain_tumour_mri')
    assert viewer.mask is None and len(viewer.region_labels) == 3
    assert not any(t.isEnabled() for t in viewer.region_toggles)


def test_run_button_is_locked_with_reason_when_checkpoint_is_missing(window, monkeypatch):
    viewer, app = window
    monkeypatch.setattr('omniseg.app.inference.checkpoint_exists', lambda task, root=None: task['id'] != 'liver_tumour_ct')
    pick_model(viewer, 'liver_tumour_ct')
    assert not viewer.infer.isEnabled()
    assert not viewer.infer_reason.isHidden() and 'Chưa có checkpoint cho model này' in viewer.infer_reason.text()
    assert 'Chưa có checkpoint' in viewer.infer.toolTip()
    monkeypatch.setattr('omniseg.app.inference.checkpoint_exists', lambda task, root=None: True)
    pick_model(viewer, 'brain_tumour_mri'); pick_model(viewer, 'liver_tumour_ct')
    assert viewer.infer.isEnabled() and viewer.infer_reason.isHidden()
