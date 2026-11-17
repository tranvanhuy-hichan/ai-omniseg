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

from desktop.app import STYLE, Workspace
from desktop.volume import Volume


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
