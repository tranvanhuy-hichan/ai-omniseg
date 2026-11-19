"""Optional headless Qt tests; lightweight CI can skip without PySide6."""
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import json
import time

import nibabel as nib
import numpy as np
import pytest

pytest.importorskip('PySide6')
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from omniseg.app import Workspace
from omniseg.core.volume import Volume
from omniseg.ui.theme import STYLE


def wait(app, condition, seconds=10):
    end = time.time() + seconds
    while not condition() and time.time() < end:
        app.processEvents(); time.sleep(0.01)
    assert condition()


def save(path, data, affine=None):
    image = nib.Nifti1Image(data, np.diag([2., 1., 3., 1.]) if affine is None else affine)
    image.header.set_xyzt_units('mm')
    nib.save(image, path)
    return path


@pytest.fixture
def window(tmp_path):
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(STYLE)
    viewer = Workspace(enable_3d=False)
    viewer.set_volume(Volume.load(save(tmp_path / 'volume.nii.gz', np.ones((10, 12, 14), dtype=np.float32))))
    viewer.show(); app.processEvents()
    yield viewer, app
    viewer.mask_unsaved = False
    viewer.close(); app.processEvents()


def pick_model(viewer, task_id):
    viewer.model_choice.setCurrentIndex(next(i for i in range(viewer.model_choice.count())
                                             if viewer.model_choice.itemData(i)['id'] == task_id))


def test_default_layout_has_four_cells_and_no_file_list(window):
    viewer, app = window
    assert not hasattr(viewer, 'cases') and not hasattr(viewer, 'library')
    assert all(cell.isVisible() for cell in viewer.cells.values())
    assert [t for t in viewer.tool_buttons] == ['locate', 'window', 'pan', 'zoom', 'ruler', 'brush', 'eraser']


def test_double_click_maximises_one_cell_and_escape_restores(window):
    viewer, app = window
    canvas = viewer.views[1].canvas
    QTest.mouseDClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(40, 40)); app.processEvents()
    assert viewer.layout_mode == '1x1' and viewer.active == 1
    assert [k for k, c in viewer.cells.items() if c.isVisible()] == [1]
    assert viewer.views[1].width() > viewer.width() // 2
    QTest.keyClick(viewer, Qt.Key.Key_Escape); app.processEvents()
    assert viewer.layout_mode == '2x2' and all(c.isVisible() for c in viewer.cells.values())
    viewer.set_layout('1x3'); app.processEvents()
    assert viewer.views[1].width() > viewer.views[2].width()      # the main cell is the large one
    QTest.mouseDClick(viewer.views[0].canvas, Qt.MouseButton.LeftButton, pos=QPoint(40, 40)); app.processEvents()
    assert viewer.layout_mode == '1x3' and viewer.main == 0       # double-click promotes a small cell
    assert viewer.views[0].width() > viewer.views[1].width()
    viewer.set_main('3d'); app.processEvents()
    assert viewer.cell3d.width() > viewer.views[2].width() and viewer.main_actions['3d'].isChecked()


def test_locate_syncs_the_other_planes_and_crosshair(window):
    viewer, app = window
    viewer.locate(2, 3, 5)                      # axial: u = x, v = y
    assert viewer.views[0].slider.value() == 3 and viewer.views[1].slider.value() == 5
    assert viewer.views[2].canvas.cursor_voxel == (3, 5)
    assert viewer.views[1].canvas.cursor_voxel == (3, viewer.views[2].slider.value())


def test_canvas_maps_clicks_to_voxels(window):
    viewer, app = window
    canvas = viewer.views[2].canvas
    point = canvas.from_plane(4.5, 6.5)
    assert canvas.voxel_at(point) == (4, 6)
    assert canvas.voxel_at(QPoint(1, 1).toPointF()) is None
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=point.toPoint())
    assert viewer.views[2].canvas.cursor_voxel == (4, 6)


def test_right_drag_changes_window_and_ruler_records_mm(window):
    viewer, app = window
    width = viewer.width_control.value()
    viewer.window_drag(50, 0)
    assert viewer.width_control.value() > width
    viewer.set_tool('ruler')
    viewer.ruler_done(2, (0.0, 0.0), (3.0, 4.0))   # spacing x=2 mm, y=1 mm in this volume
    assert '7.2 mm' in viewer.ruler_list.item(0).text()   # hypot(6, 4)
    assert viewer.views[2].canvas.rulers
    viewer.clear_rulers(); assert viewer.ruler_list.count() == 0


def test_region_panel_follows_selected_task(window):
    viewer, app = window
    pick_model(viewer, 'brain_tumour_mri')
    assert viewer.label_ids == [1, 2, 3] and 'BRAIN TUMOUR' in viewer.region_title.text()
    assert viewer.brush_label.count() == 3
    pick_model(viewer, 'liver_tumour_ct')
    assert viewer.label_ids == [1, 2] and 'LIVER' in viewer.region_title.text() and viewer.brush_label.count() == 2


def test_model_is_suggested_from_channel_count(window, tmp_path):
    viewer, app = window                         # one channel, not CT-like values -> a 1-channel task
    assert viewer.current_task['id'] == 'liver_tumour_ct'
    viewer.set_volume(Volume.load(save(tmp_path / 'four.nii.gz', np.ones((6, 6, 6, 4), dtype=np.float32))))
    assert viewer.current_task['id'] == 'brain_tumour_mri'


def test_mask_amounts_and_model_switch_drops_mask(window, tmp_path):
    viewer, app = window
    pick_model(viewer, 'liver_tumour_ct')
    mask = np.zeros((10, 12, 14), dtype=np.uint8); mask[2:5, 2:5, 2:5] = 1; mask[3, 3, 3] = 2
    viewer.set_mask(mask, tmp_path / 'mask.nii.gz')
    assert '26 voxel' in viewer.region_labels[0].text() and '1 voxel' in viewer.region_labels[1].text()
    wait(app, lambda: viewer.analysis is not None)
    assert len(viewer.stats_rows) == 2 and viewer.stats_rows[0]['Số khối rời nhau'] == '1'
    pick_model(viewer, 'brain_tumour_mri')
    assert viewer.mask is None and not any(t.isEnabled() for t in viewer.region_toggles)


def test_clicking_a_region_selects_the_brush_label_and_tool_shows_options(window):
    viewer, app = window
    pick_model(viewer, 'liver_tumour_ct')
    assert viewer.brush_box.isHidden()
    viewer.set_tool('brush'); app.processEvents()
    assert not viewer.brush_box.isHidden() and viewer.tabs.count() == 4
    QTest.mouseClick(viewer.region_widgets[1], Qt.MouseButton.LeftButton)
    assert viewer.brush_label.currentData() == 2
    assert 'background' in viewer.region_widgets[1].styleSheet() and not viewer.region_widgets[0].styleSheet()
    viewer.set_tool('locate'); assert viewer.brush_box.isHidden() and not viewer.region_widgets[1].styleSheet()


def test_brush_creates_mask_undo_redo_and_eraser(window):
    viewer, app = window
    viewer.set_tool('brush'); viewer.brush_size.setValue(3.0)
    viewer.views[2].slider.setValue(4)
    viewer.paint(2, 5, 5, 'start'); viewer.paint(2, 8, 5, 'move'); viewer.paint(2, 0, 0, 'end')
    assert viewer.mask is not None and viewer.mask_kind == 'drawn' and viewer.edited
    painted = int(np.count_nonzero(viewer.mask)); assert painted > 0
    assert viewer.mask[:, :, 4].any() and not viewer.mask[:, :, 3].any()
    assert 'ĐÃ CHỈNH SỬA' in viewer.views[2].canvas.banner
    viewer.undo(); assert not viewer.mask.any()
    viewer.redo(); assert np.count_nonzero(viewer.mask) == painted
    viewer.set_tool('eraser')
    viewer.paint(2, 5, 5, 'start'); viewer.paint(2, 0, 0, 'end')
    assert np.count_nonzero(viewer.mask) < painted
    wait(app, lambda: viewer.analysis is not None)


def test_reference_comparison_fills_dice_table(window, tmp_path):
    viewer, app = window
    pick_model(viewer, 'liver_tumour_ct')
    mask = np.zeros((10, 12, 14), dtype=np.uint8); mask[:4, :4, :4] = 1
    viewer.set_mask(mask, tmp_path / 'a.nii.gz')
    ref = mask.copy(); ref[:2] = 0
    viewer.set_reference(ref, tmp_path / 'ref.nii.gz')
    wait(app, lambda: viewer.analysis is not None and viewer.analysis['comparison'] is not None)
    assert viewer.compare_rows[0]['Dice'] == '0.667'
    assert viewer.compare_rows[1]['Dice'] == 'n/a'      # label 2 empty in both


def test_save_mask_and_report(window, tmp_path):
    viewer, app = window
    mask = np.zeros((10, 12, 14), dtype=np.uint8); mask[1:3, 1:3, 1:3] = 1
    viewer.set_mask(mask, None, kind='drawn'); assert viewer.mask_unsaved
    out = viewer.write_mask(tmp_path / 'out.nii.gz')
    assert out.is_file() and not viewer.mask_unsaved
    assert np.array_equal(nib.load(out).get_fdata().astype(np.uint8), mask)
    wait(app, lambda: viewer.analysis is not None)
    report = viewer.write_report(str(tmp_path / 'r.json'))
    assert json.loads(open(report, encoding='utf-8').read())['regions'][0]['voxels'] == 8


def test_run_button_is_locked_with_reason_when_checkpoint_is_missing(window, monkeypatch):
    viewer, app = window
    monkeypatch.setattr('omniseg.app.inference.checkpoint_exists', lambda task, root=None: task['id'] != 'liver_tumour_ct')
    pick_model(viewer, 'brain_tumour_mri'); pick_model(viewer, 'liver_tumour_ct')
    assert not viewer.infer.isEnabled()
    assert not viewer.infer_reason.isHidden() and 'Chưa có checkpoint cho model này' in viewer.infer_reason.text()
    monkeypatch.setattr('omniseg.app.inference.checkpoint_exists', lambda task, root=None: True)
    pick_model(viewer, 'brain_tumour_mri'); pick_model(viewer, 'liver_tumour_ct')
    assert viewer.infer.isEnabled() and viewer.infer_reason.isHidden()
