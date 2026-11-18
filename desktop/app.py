"""Qt Widgets workspace for MRI slice review and reference-mask overlays."""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QSize, QTimer
from PySide6.QtGui import QAction, QColor, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame,
    QGridLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QSlider, QSplitter, QVBoxLayout, QWidget,
    QLineEdit, QGroupBox, QScrollArea, QStackedWidget,
)

from .volume import COLORS, Volume, render_slice

ROOT = Path(__file__).resolve().parents[1]
ASSETS = (Path(__file__).resolve().parent / 'assets').as_posix()
# This pilot checkpoint (fold 0, trained the full 100/100 epochs) is the only
# one trained so far; point these at a newer run's dataset/fold/checkpoint
# when one is trained.
INFER_DATASET_ID = 501
INFER_FOLD = 0
INFER_TRAINER = 'nnUNetTrainer_100epochs'
INFER_CHECKPOINT = 'checkpoint_final.pth'
INFER_ENV = {
    'nnUNet_raw': str(ROOT / 'data/nnUNet_raw'),
    'nnUNet_preprocessed': str(ROOT / 'data/nnUNet_preprocessed'),
    'nnUNet_results': str(ROOT / 'artifacts/nnUNet_results'),
}
STYLE = '''
QMainWindow, QWidget { background: #f3f6fa; color: #24354a; font-family: "Noto Sans", "Segoe UI", sans-serif; font-size: 14px; }
QLabel { background: transparent; }
QFrame#slicePanel QSlider { background: transparent; outline: none; }
QFrame#slicePanel QSlider:focus { outline: none; }
QFrame#slicePanel QSlider::groove:horizontal { background: #28384b; height: 4px; border-radius: 2px; }
QFrame#slicePanel QSlider::sub-page:horizontal { background: #148c99; height: 4px; border-radius: 2px; }
QFrame#slicePanel QSlider::add-page:horizontal { background: #28384b; height: 4px; border-radius: 2px; }
QFrame#slicePanel QSlider::handle:horizontal { width: 12px; height: 12px; margin: -4px 0; border-radius: 6px; background: #148c99; border: none; }
QFrame#slicePanel QSlider::handle:horizontal:hover { background: #3fb7c0; }
QFrame#slicePanel QPushButton { background: #172638; color: #c8d7e5; border: 1px solid #35485d; padding: 5px 9px; font-size: 12px; }
QFrame#slicePanel QPushButton:hover { background: #253c50; color: white; }
QPushButton#expandButton { padding: 0; border-radius: 13px; font-size: 14px; font-weight: 700; }
QPushButton#sliderStepButton { background: #1d3042; color: #d7e3ec; border: 1px solid #35485d; padding: 0; border-radius: 11px; font-size: 15px; font-weight: 700; }
QPushButton#sliderStepButton:hover { background: #148c99; color: white; border-color: #148c99; }
QPushButton#sliderStepButton:pressed { background: #0e7178; }
QPushButton#sliderStepButton:disabled { background: #172638; color: #44566a; border-color: #223247; }
QScrollArea { border: none; }
QScrollBar:vertical { width: 9px; background: #edf1f5; }
QScrollBar::handle:vertical { background: #bbc9d6; min-height: 35px; border-radius: 4px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QFrame#header { background: white; border-bottom: 1px solid #dae3ec; }
QLabel#brand { color: #087e8b; font-size: 18px; font-weight: 700; }
QLabel#title { font-size: 21px; font-weight: 650; }
QLabel#muted { color: #748399; font-size: 13px; }
QLabel#eyebrow { color: #748399; font-size: 11px; font-weight: 700; }
QLabel#badge { background: #e5f4f1; color: #13786f; padding: 6px 10px; border-radius: 5px; font-size: 11px; }
QPushButton { background: white; border: 1px solid #d6e0ea; padding: 9px 13px; border-radius: 5px; font-weight: 600; }
QPushButton:hover { background: #edf5f8; border-color: #92bac6; }
QPushButton#primary { background: #087e8b; color: white; border: 1px solid #087e8b; }
QPushButton#primary:hover { background: #066b76; }
QPushButton:disabled { background: #e9eef4; border-color: #dce4ed; color: #98a5b6; }
QLineEdit, QComboBox, QDoubleSpinBox { background: white; border: 1px solid #d7e1ec; padding: 5px 7px; border-radius: 6px; selection-background-color: #cdeae8; }
QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus { border-color: #45a6a8; }
QComboBox:hover, QDoubleSpinBox:hover { border-color: #b6cad8; }
QComboBox::drop-down { width: 26px; border: none; }
QComboBox::down-arrow { image: url(ASSETS_DIR/chevron_down.svg); width: 11px; height: 11px; }
QComboBox QAbstractItemView { background: white; border: 1px solid #dce5ee; border-radius: 6px; outline: none; padding: 4px; selection-background-color: #e1f0f2; selection-color: #076d78; }
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 18px; border: none; background: transparent; }
QDoubleSpinBox::up-arrow { image: url(ASSETS_DIR/chevron_up.svg); width: 9px; height: 9px; }
QDoubleSpinBox::down-arrow { image: url(ASSETS_DIR/chevron_down.svg); width: 9px; height: 9px; }
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover { background: #eef3f8; }
QListWidget { background: #f9fbfd; border: 1px solid #dce5ee; border-radius: 6px; outline: none; }
QListWidget::item { padding: 10px 10px; border-bottom: 1px solid #e9eef4; border-radius: 4px; }
QListWidget::item:hover { background: #eef5f7; }
QListWidget::item:selected { background: #e1f0f2; color: #076d78; border-left: 3px solid #087e8b; font-weight: 600; }
QFrame#libraryFooter { background: #f3f6fa; border-radius: 6px; border: none; }
QFrame#inferBox { background: #e5f4f1; border: 1px solid #bfe3dc; border-radius: 8px; }
QFrame#inferBox QPushButton#primary { font-size: 14px; padding: 11px 13px; }
QProgressBar#inferProgress { background: #bfe3dc; border: none; border-radius: 2px; }
QProgressBar#inferProgress::chunk { background: #087e8b; border-radius: 2px; }
QGroupBox { background: white; border: 1px solid #dce5ee; border-radius: 8px; margin-top: 12px; padding: 16px 13px 13px; font-weight: 600; font-size: 13px; color: #42566d; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; color: #1c2c3f; font-size: 13px; }
QGroupBox QLabel, QGroupBox QCheckBox { background: white; }
QGroupBox QLabel#muted { font-weight: 400; }
QCheckBox { spacing: 10px; }
QCheckBox::indicator { width: 36px; height: 20px; border: none; }
QCheckBox::indicator:unchecked { image: url(ASSETS_DIR/switch_off.svg); }
QCheckBox::indicator:checked { image: url(ASSETS_DIR/switch_on.svg); }
QCheckBox::indicator:disabled { image: url(ASSETS_DIR/switch_disabled.svg); }
QSlider::groove:horizontal { height: 4px; background: #dbe4ed; border-radius: 2px; }
QSlider::sub-page:horizontal { height: 4px; background: #45a6a8; border-radius: 2px; }
QSlider::handle:horizontal { width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; background: #148c99; border: 2px solid white; }
QSlider::handle:horizontal:hover { background: #0e8f99; }
QSlider::handle:horizontal:pressed { background: #0a747d; }
QStatusBar { background: white; border-top: 1px solid #dce5ee; color: #748399; }
QSplitter::handle { background: #e0e7ef; }
'''.replace('ASSETS_DIR', ASSETS)


def label(text, name=None):
    widget = QLabel(text)
    if name:
        widget.setObjectName(name)
    widget.setWordWrap(True)
    return widget


class Spinner(QWidget):
    """Small indeterminate spinner overlaid while a volume/mask loads in the background."""

    def __init__(self, parent=None, diameter=48):
        super().__init__(parent)
        self.diameter = diameter
        self.angle = 0
        self.setFixedSize(diameter, diameter)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance)
        self.hide()

    def _advance(self):
        self.angle = (self.angle + 30) % 360
        self.update()

    def start(self):
        if self.parent() is not None:
            parent_rect = self.parent().rect()
            self.move((parent_rect.width()-self.width())//2, (parent_rect.height()-self.height())//2)
        self.show(); self.raise_(); self.timer.start(80)

    def stop(self):
        self.timer.stop(); self.hide()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(4, 4, -4, -4)
        pen_width = max(3, self.diameter // 12)
        for i in range(12):
            fade = (i / 12)
            color = QColor('#148c99')
            color.setAlphaF(max(0.08, 1 - fade))
            pen = painter.pen(); pen.setColor(color); pen.setWidth(pen_width); pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            start_angle = (self.angle - i * 30) * 16
            painter.drawArc(rect, start_angle, 16 * 18)


class Loader(QThread):
    loaded = Signal(object)
    failed = Signal(str)

    def __init__(self, path, volume=None):
        super().__init__()
        self.path, self.volume = path, volume

    def run(self):
        try:
            self.loaded.emit(self.volume.load_mask(self.path) if self.volume else Volume.load(self.path))
        except Exception as error:
            self.failed.emit(str(error))


class InferenceWorker(QThread):
    """Runs nnUNetv2_predict on the open volume's 4 channels in a temp folder."""
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, volume):
        super().__init__()
        self.volume = volume

    def run(self):
        try:
            with tempfile.TemporaryDirectory(prefix='neuroseg-infer-') as tmp:
                tmp = Path(tmp)
                input_dir, output_dir = tmp / 'input', tmp / 'output'
                input_dir.mkdir(); output_dir.mkdir()
                case = self.volume.path.name.removesuffix('.nii.gz').removesuffix('.nii')
                data, affine = self.volume.data, self.volume.image.affine
                for channel in range(data.shape[3]):
                    nib.save(nib.Nifti1Image(data[..., channel], affine), input_dir / f'{case}_{channel:04d}.nii.gz')
                env = dict(os.environ)
                for key, value in INFER_ENV.items():
                    env.setdefault(key, value)
                subprocess.run(
                    ['nnUNetv2_predict', '-d', str(INFER_DATASET_ID), '-c', '2d', '-f', str(INFER_FOLD),
                     '-i', str(input_dir), '-o', str(output_dir), '-tr', INFER_TRAINER, '-chk', INFER_CHECKPOINT],
                    env=env, check=True, capture_output=True, text=True)
                predicted = output_dir / f'{case}.nii.gz'
                if not predicted.is_file():
                    raise FileNotFoundError('Không tìm thấy mask dự đoán sau khi chạy xong.')
                destination = Path(tempfile.gettempdir()) / f'neuroseg-prediction-{case}.nii.gz'
                destination.write_bytes(predicted.read_bytes())
                self.done.emit(str(destination))
        except subprocess.CalledProcessError as error:
            self.failed.emit((error.stderr or str(error))[-2000:])
        except Exception as error:
            self.failed.emit(str(error))


class SliceCanvas(QWidget):
    step = Signal(int)
    expand = Signal()

    def __init__(self, axis):
        super().__init__()
        self.axis, self.rgb, self.spacing = axis, None, (1, 1)
        self.setMinimumSize(180, 180)
        self.setSizePolicy(self.sizePolicy().Policy.Expanding, self.sizePolicy().Policy.Expanding)
        self.spinner = Spinner(self, diameter=32)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.spinner.move((self.width()-self.spinner.width())//2, (self.height()-self.spinner.height())//2)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor('#101c2a'))
        if self.rgb is None:
            painter.setPen(QColor('#718296'))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'Mở MRI để xem lát cắt')
            return
        data = self.rgb
        image = QImage(data.data, data.shape[1], data.shape[0], data.strides[0], QImage.Format.Format_RGB888)
        physical = QSize(max(1, round(data.shape[1] * self.spacing[0] * 10)), max(1, round(data.shape[0] * self.spacing[1] * 10)))
        fitted = physical.scaled(max(1, self.width()-44), max(1, self.height()-44), Qt.AspectRatioMode.KeepAspectRatio)
        x, y = (self.width()-fitted.width())//2, (self.height()-fitted.height())//2
        from PySide6.QtCore import QRect
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(QRect(x, y, fitted.width(), fitted.height()), image)
        left, right, top, bottom = {2: ('L', 'R', 'A', 'P'), 1: ('L', 'R', 'S', 'I'), 0: ('P', 'A', 'S', 'I')}[self.axis]
        painter.setPen(QColor('#8ba6b8'))
        painter.drawText(9, self.height()//2, left)
        painter.drawText(self.width()-20, self.height()//2, right)
        painter.drawText(self.width()//2, 16, top)
        painter.drawText(self.width()//2, self.height()-8, bottom)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.expand.emit()
            event.accept()

    def wheelEvent(self, event):
        self.step.emit(1 if event.angleDelta().y() > 0 else -1)
        event.accept()


class SlicePanel(QFrame):
    changed = Signal(int, int)
    expand = Signal(int)

    def __init__(self, title, axis):
        super().__init__()
        self.axis = axis
        self.setObjectName("slicePanel")
        self.setStyleSheet('SlicePanel { background: #101c2a; border-radius: 6px; }')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        row = QHBoxLayout()
        heading = label(title.split(" · ")[0])
        heading.setToolTip(title)
        heading.setStyleSheet('background: transparent; color: #b9cad8; font-weight: 600;')
        self.counter = label('—')
        self.counter.setStyleSheet('background: transparent; color: #8da2b7;')
        heading.setWordWrap(False)
        self.counter.setWordWrap(False)
        self.expand_button = QPushButton('⤢')
        self.expand_button.setObjectName('expandButton')
        self.expand_button.setFixedSize(26, 26)
        self.expand_button.setToolTip('Phóng mặt phẳng ra toàn vùng xem · Nhấp đúp ảnh để chuyển nhanh · Esc để quay lại')
        self.expand_button.clicked.connect(lambda: self.expand.emit(axis))
        row.addWidget(heading); row.addStretch(); row.addWidget(self.counter); row.addSpacing(8); row.addWidget(self.expand_button)
        layout.addLayout(row)
        self.canvas = SliceCanvas(axis)
        self.canvas.expand.connect(lambda: self.expand.emit(axis))
        self.canvas.setToolTip("Cuộn: chuyển lát cắt · Nhấp đúp: phóng lớn / thu lại")
        layout.addWidget(self.canvas, 1)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.slider.valueChanged.connect(lambda value: self.changed.emit(axis, value))
        self.canvas.step.connect(lambda step: self.slider.setValue(self.slider.value()+step))
        slider_row = QHBoxLayout(); slider_row.setContentsMargins(0, 0, 0, 0); slider_row.setSpacing(6)
        self.prev_button = QPushButton('‹'); self.prev_button.setObjectName('sliderStepButton'); self.prev_button.setFixedSize(22, 22)
        self.prev_button.setEnabled(False); self.prev_button.setToolTip('Lát cắt trước')
        self.prev_button.clicked.connect(lambda: self.slider.setValue(self.slider.value()-1))
        self.next_button = QPushButton('›'); self.next_button.setObjectName('sliderStepButton'); self.next_button.setFixedSize(22, 22)
        self.next_button.setEnabled(False); self.next_button.setToolTip('Lát cắt kế tiếp')
        self.next_button.clicked.connect(lambda: self.slider.setValue(self.slider.value()+1))
        slider_row.addWidget(self.prev_button); slider_row.addWidget(self.slider, 1); slider_row.addWidget(self.next_button)
        layout.addLayout(slider_row)


class Workspace(QMainWindow):
    def __init__(self):
        super().__init__()
        self.volume = self.mask = self.worker = None
        self.mask_path = None
        self.loading = False
        self.focus_axis = None
        self.surface_viewer = None
        self.setWindowTitle('NeuroSeg · Không gian phân đoạn')
        self.setWindowIcon(QIcon(f'{ASSETS}/logo.svg'))
        self.resize(1540, 940)
        self.setMinimumSize(1140, 740)
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        header = QFrame(); header.setObjectName('header'); header.setFixedHeight(48)
        bar = QHBoxLayout(header); bar.setContentsMargins(16, 4, 16, 4)
        brand_box = QWidget(); brand_layout = QHBoxLayout(brand_box); brand_layout.setContentsMargins(0, 0, 0, 0); brand_layout.setSpacing(8)
        logo = QLabel(); logo.setPixmap(QPixmap(f'{ASSETS}/logo.svg').scaled(24, 24, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        brand_layout.addWidget(logo); brand_layout.addWidget(label('NeuroSeg', 'brand'))
        bar.addWidget(brand_box); bar.addSpacing(18)
        subtitle = label('KHÔNG GIAN PHÂN ĐOẠN', 'eyebrow'); subtitle.setWordWrap(False)
        subtitle.hide(); bar.addStretch()
        bar.addWidget(label('Viewer nghiên cứu', 'muted')); bar.addSpacing(12)
        self.open_button = QPushButton('＋  Mở MRI'); self.open_button.setObjectName('primary'); self.open_button.clicked.connect(self.open_file)
        self.mask_button = QPushButton('Nạp mask'); self.mask_button.clicked.connect(self.open_mask); self.mask_button.setEnabled(False)
        bar.addWidget(self.mask_button); bar.addWidget(self.open_button)
        root.addWidget(header)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter = splitter
        root.addWidget(splitter, 1)
        left = QWidget(); left.setObjectName('library'); self.library = left; left.setMinimumWidth(240); left.setMaximumWidth(320)
        left_layout = QVBoxLayout(left); left_layout.setContentsMargins(16, 18, 14, 16); left_layout.setSpacing(12)
        header_box = QVBoxLayout(); header_box.setSpacing(2)
        header_box.addWidget(label('THƯ VIỆN', 'eyebrow'))
        title_row = QHBoxLayout(); title_row.setContentsMargins(0, 0, 0, 0)
        case_title = label('Danh sách ca'); case_title.setStyleSheet('font-size: 15px; font-weight: 700;')
        title_row.addWidget(case_title); title_row.addStretch()
        self.case_count = label('0 ca', 'badge'); self.case_count.setObjectName('badge'); title_row.addWidget(self.case_count)
        header_box.addLayout(title_row)
        left_layout.addLayout(header_box)
        self.search = QLineEdit(); self.search.setPlaceholderText('🔍  Tìm theo tên file…'); self.search.textChanged.connect(self.filter_cases); left_layout.addWidget(self.search)
        folder = QPushButton('📁  Chọn thư mục MRI'); folder.setObjectName('primary'); folder.clicked.connect(self.open_folder); left_layout.addWidget(folder)
        self.cases = QListWidget(); self.cases.itemActivated.connect(self.open_item); self.cases.itemClicked.connect(self.open_item); left_layout.addWidget(self.cases, 1)
        footer_note = QFrame(); footer_note.setObjectName('libraryFooter')
        footer_note_layout = QVBoxLayout(footer_note); footer_note_layout.setContentsMargins(10, 8, 10, 8)
        footer_note_layout.addWidget(label('Tệp NIfTI · .nii / .nii.gz\nDữ liệu được đọc trên máy của bạn.', 'muted'))
        left_layout.addWidget(footer_note)
        splitter.addWidget(left)
        middle = QWidget(); mid = QVBoxLayout(middle); mid.setContentsMargins(10, 4, 10, 8); mid.setSpacing(6)
        self.workspace_layout = mid
        self.case_title = label('Chưa mở MRI')
        self.case_title.setStyleSheet('font-size: 15px; font-weight: 700;')
        self.case_title.setWordWrap(False)
        self.case_subtitle = label(''); self.case_subtitle.hide()
        controls_widget = QWidget(); controls_widget.setFixedHeight(42)
        controls = QHBoxLayout(controls_widget); controls.setContentsMargins(0, 0, 0, 0); controls.setSpacing(8)
        controls.addWidget(self.case_title); controls.addSpacing(10)
        self.channel = QComboBox(); self.channel.setMinimumWidth(130); self.channel.setEnabled(False); self.channel.currentIndexChanged.connect(self.channel_changed)
        controls.addWidget(self.channel); controls.addSpacing(6)
        controls.addWidget(label('Bố cục'))
        self.layout_choice = QComboBox()
        for title, axis in [('Ba mặt phẳng', None), ('Axial · Toàn vùng xem', 2), ('Coronal · Toàn vùng xem', 1), ('Sagittal · Toàn vùng xem', 0), ('3D · Não và mask', '3d')]:
            self.layout_choice.addItem(title, axis)
        self.layout_choice.currentIndexChanged.connect(lambda index: self.set_focus(self.layout_choice.itemData(index)))
        controls.addWidget(self.layout_choice); controls.addStretch()
        reset = QPushButton('Đặt lại W/L'); reset.clicked.connect(self.reset_window); controls.addWidget(reset)
        mid.addWidget(controls_widget)
        self.viewer_stack = QStackedWidget()
        self.viewer = QWidget()
        self.focus_viewer = QWidget()
        self.focus_layout = QVBoxLayout(self.focus_viewer)
        self.focus_layout.setContentsMargins(0, 0, 0, 0)
        self.viewer_stack.addWidget(self.viewer)
        self.viewer_stack.addWidget(self.focus_viewer)
        grid = QGridLayout(self.viewer); grid.setContentsMargins(0, 0, 0, 0); grid.setSpacing(10)
        self.viewer_grid = grid
        self.panels = []
        for index, (title, axis) in enumerate([('AXIAL · Trục ngang', 2), ('CORONAL · Trán', 1), ('SAGITTAL · Dọc', 0)]):
            panel = SlicePanel(title, axis); panel.changed.connect(self.slice_changed)
            panel.expand.connect(self.toggle_focus)
            self.panels.append(panel)
        # Main plane on the left, two companion planes on the right.
        grid.addWidget(self.panels[0], 0, 0, 2, 1)
        grid.addWidget(self.panels[1], 0, 1)
        grid.addWidget(self.panels[2], 1, 1)
        overview = QGroupBox('Kết quả phân đoạn'); info = QVBoxLayout(overview); info.setSpacing(10)
        info.addWidget(label('VÙNG TỔN THƯƠNG · MSD TASK01', 'eyebrow'))
        self.region_labels = []
        self.region_toggles = []
        for name, color in zip(['Phù nề', 'U không tăng tương phản', 'U tăng tương phản'], COLORS[1:]):
            row_widget = QWidget(); row_layout = QHBoxLayout(row_widget); row_layout.setContentsMargins(0, 0, 0, 0); row_layout.setSpacing(0)
            dot = label('●'); dot.setStyleSheet(f'color: rgb({color[0]}, {color[1]}, {color[2]}); font-size: 15px;'); dot.setFixedWidth(16)
            text_widget = QWidget(); text_layout = QVBoxLayout(text_widget); text_layout.setContentsMargins(6, 0, 0, 0); text_layout.setSpacing(1)
            name_label = label(name); name_label.setStyleSheet('font-weight: 600;')
            amount = label('Chưa có mask', 'muted')
            text_layout.addWidget(name_label); text_layout.addWidget(amount)
            toggle = QCheckBox(); toggle.setChecked(True); toggle.setEnabled(False)
            toggle.toggled.connect(self.refresh)
            row_layout.addWidget(dot, 0, Qt.AlignmentFlag.AlignTop); row_layout.addWidget(text_widget, 1)
            row_layout.addWidget(toggle, 0, Qt.AlignmentFlag.AlignVCenter)
            self.region_labels.append(amount)
            self.region_toggles.append(toggle)
            info.addWidget(row_widget)
        self.mask_source = label('Chưa nạp mask. Màu phủ chỉ xuất hiện khi có nhãn tham chiếu hoặc mask bạn mở.', 'muted')
        self.mask_source.setStyleSheet('background: #f3f6fa; border-radius: 4px; padding: 8px; color: #556377; font-size: 12px;')
        info.addWidget(self.mask_source)

        grid.setRowStretch(0, 1); grid.setRowStretch(1, 1); grid.setColumnStretch(0, 3); grid.setColumnStretch(1, 2)
        mid.addWidget(self.viewer_stack, 1)
        self.view_hint = label('Nhấp đúp ảnh hoặc bấm Phóng lớn để xem một mặt phẳng · Esc để quay lại', 'muted')
        self.view_hint.hide(); splitter.addWidget(middle)
        right = QWidget(); right_layout = QVBoxLayout(right); right_layout.setContentsMargins(14, 18, 14, 16); right_layout.setSpacing(14)
        right_scroll = QScrollArea(); right_scroll.setWidgetResizable(True); right_scroll.setWidget(right)
        right_scroll.setMinimumWidth(320); right_scroll.setMaximumWidth(420)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.inspector = right_scroll
        self.inspector_layout = right_layout
        right_layout.addWidget(label('BẢNG ĐIỀU KHIỂN', 'eyebrow'))
        infer_box = QFrame(); infer_box.setObjectName('inferBox')
        infer_layout = QVBoxLayout(infer_box); infer_layout.setContentsMargins(12, 12, 12, 12); infer_layout.setSpacing(8)
        self.infer = QPushButton('▶  Chạy phân đoạn'); self.infer.setObjectName('primary'); self.infer.setEnabled(False)
        self.infer.setToolTip('Chạy checkpoint đã train (100/100 epoch, fold 0) trên ảnh đang mở.')
        self.infer.clicked.connect(self.run_inference)
        self.inference_worker = None
        infer_layout.addWidget(self.infer)
        self.infer_progress = QProgressBar(); self.infer_progress.setObjectName('inferProgress')
        self.infer_progress.setRange(0, 0); self.infer_progress.setTextVisible(False); self.infer_progress.setFixedHeight(4)
        self.infer_progress.hide()
        infer_layout.addWidget(self.infer_progress)
        self.infer_dots_timer = QTimer(self); self.infer_dots_timer.setInterval(450)
        self.infer_dots_timer.timeout.connect(self._animate_infer_text)
        self.infer_dots_count = 0
        infer_note = label('Viewer nghiên cứu · chưa có kết quả AI hoặc mô hình được tích hợp. Không dùng để đưa ra chẩn đoán.', 'muted')
        infer_layout.addWidget(infer_note)
        right_layout.addWidget(infer_box)
        display = QGroupBox('Hiển thị MRI'); self.slice_display_group = display; display_layout = QVBoxLayout(display); display_layout.setSpacing(6)
        windowing = QGridLayout(); windowing.setHorizontalSpacing(10); windowing.setVerticalSpacing(2)
        self.level = QDoubleSpinBox(); self.width_control = QDoubleSpinBox()
        for control in (self.level, self.width_control):
            control.setRange(-1e6, 1e6); control.setDecimals(1); control.setEnabled(False); control.valueChanged.connect(self.refresh)
        self.width_control.setMinimum(0.1)
        windowing.addWidget(label('Window level', 'muted'), 0, 0); windowing.addWidget(label('Window width', 'muted'), 0, 1)
        windowing.addWidget(self.level, 1, 0); windowing.addWidget(self.width_control, 1, 1)
        windowing.setColumnStretch(0, 1); windowing.setColumnStretch(1, 1)
        display_layout.addLayout(windowing)
        display_layout.addWidget(label('Cuộn chuột hoặc kéo thanh dưới mỗi ảnh để chuyển lát cắt.', 'muted'))
        right_layout.addWidget(display)
        mask_group = QGroupBox('Lớp phủ lát cắt'); self.slice_mask_group = mask_group; mask_layout = QVBoxLayout(mask_group); mask_layout.setSpacing(8)
        show_mask_row = QHBoxLayout(); show_mask_row.setContentsMargins(0, 0, 0, 0)
        show_mask_row.addWidget(label('Hiện mask')); show_mask_row.addStretch()
        self.show_mask = QCheckBox(); self.show_mask.setChecked(True); self.show_mask.setEnabled(False); self.show_mask.toggled.connect(self.refresh)
        show_mask_row.addWidget(self.show_mask)
        mask_layout.addLayout(show_mask_row)
        opacity_row = QHBoxLayout(); opacity_row.setContentsMargins(0, 0, 0, 0)
        opacity_row.addWidget(label('Độ phủ', 'muted'))
        self.opacity = QSlider(Qt.Orientation.Horizontal); self.opacity.setRange(0, 100); self.opacity.setValue(35); self.opacity.valueChanged.connect(self.refresh)
        opacity_row.addWidget(self.opacity, 1)
        mask_layout.addLayout(opacity_row)
        self.clear_mask_button = QPushButton('Bỏ mask'); self.clear_mask_button.setEnabled(False); self.clear_mask_button.clicked.connect(self.clear_mask)
        mask_layout.addWidget(self.clear_mask_button)
        right_layout.addWidget(mask_group)
        right_layout.addWidget(overview)
        details = QGroupBox('Thông tin thể tích'); details_layout = QVBoxLayout(details)
        self.details = label('Chưa mở MRI.', 'muted'); self.details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse); details_layout.addWidget(self.details); right_layout.addWidget(details)
        right_layout.addStretch()
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 0); splitter.setStretchFactor(1, 1); splitter.setStretchFactor(2, 0)
        splitter.setSizes([220, 900, 360]); splitter.setCollapsible(0, False); splitter.setCollapsible(1, False); splitter.setCollapsible(2, False)
        self.statusBar().showMessage('Sẵn sàng · Mở NIfTI 3D/4D · Định hướng neurological sau khi chuyển về RAS')
        action = QAction('Mở MRI', self); action.setShortcut('Ctrl+O'); action.triggered.connect(self.open_file); self.addAction(action)
        escape = QAction('Trở lại ba mặt phẳng', self); escape.setShortcut('Esc'); escape.triggered.connect(lambda: self.set_focus(None)); self.addAction(escape)
        default = ROOT / 'data/source/Task01_BrainTumour/imagesTr'
        if default.is_dir(): self.populate(default)

    def toggle_focus(self, axis):
        self.set_focus(None if self.focus_axis == axis else axis)

    def set_focus(self, axis):
        if axis == '3d':
            self.show_3d()
            return
        if axis is not None and axis not in (0, 1, 2):
            return
        # Separate pages make overview and focus mutually exclusive. Moving the
        # actual panel preserves its slider, buffer and mask state.
        self.focus_axis = axis
        self.slice_display_group.show(); self.slice_mask_group.show()
        if self.surface_viewer: self.surface_viewer.settings.hide()
        for index, panel in enumerate(self.panels):
            self.focus_layout.removeWidget(panel)
            self.viewer_grid.removeWidget(panel)
            if index == 0:
                self.viewer_grid.addWidget(panel, 0, 0, 2, 1)
            else:
                self.viewer_grid.addWidget(panel, index-1, 1)
            panel.expand_button.setText('↙' if panel.axis == axis else '⤢')
            panel.show()
        if axis is None:
            self.viewer_stack.setCurrentWidget(self.viewer)
        else:
            selected = next(p for p in self.panels if p.axis == axis)
            self.viewer_grid.removeWidget(selected)
            self.focus_layout.addWidget(selected)
            selected.show()
            self.viewer_stack.setCurrentWidget(self.focus_viewer)
        self.layout_choice.blockSignals(True)
        self.layout_choice.setCurrentIndex(self.layout_choice.findData(axis))
        self.layout_choice.blockSignals(False)
        self.statusBar().showMessage('Ba mặt phẳng · Nhấp đúp một ảnh để phóng lớn' if axis is None
                                    else 'Một mặt phẳng chiếm toàn bộ vùng ảnh · Nhấn Esc hoặc Thu lại để quay lại')

    def show_3d(self):
        if self.surface_viewer is None:
            try:
                from .view3d import SurfaceViewer
                self.surface_viewer = SurfaceViewer()
                self.viewer_stack.addWidget(self.surface_viewer)
                self.inspector_layout.insertWidget(1,self.surface_viewer.settings)
                self.update_3d_input()
            except ImportError as error:
                self.set_focus(None)
                QMessageBox.warning(self, 'Thiếu thư viện 3D', f'Chạy make setup-desktop để cài VTK.\n{error}')
                return
        self.focus_axis = '3d'
        self.surface_viewer.settings.show()
        self.slice_display_group.hide(); self.slice_mask_group.hide()
        self.viewer_stack.setCurrentWidget(self.surface_viewer)
        self.layout_choice.blockSignals(True)
        self.layout_choice.setCurrentIndex(self.layout_choice.findData('3d'))
        self.layout_choice.blockSignals(False)
        self.surface_viewer.activate()
        self.statusBar().showMessage('3D từ mask · Chuột trái: xoay · Cuộn: zoom · Esc: trở về lát cắt')

    def update_3d_input(self):
        if self.surface_viewer is None: return
        self.surface_viewer.set_anatomy(self.volume,max(0,self.channel.currentIndex()))
        if self.volume is None or self.mask is None:
            self.surface_viewer.set_input(None)
        else:
            self.surface_viewer.set_input(self.mask, self.volume.image.affine, self.mask_source.text())

    def populate(self, folder):
        self.cases.clear()
        paths = sorted(p for p in Path(folder).iterdir() if p.is_file() and p.name.endswith(('.nii', '.nii.gz')) and not p.name.startswith('._'))
        for path in paths:
            item = QListWidgetItem(path.name.removesuffix('.nii.gz').removesuffix('.nii'))
            item.setData(Qt.ItemDataRole.UserRole, str(path)); item.setToolTip(path.name); self.cases.addItem(item)
        self.case_count.setText(f'{len(paths)} ca')
        self.filter_cases(self.search.text())

    def filter_cases(self, query):
        for i in range(self.cases.count()):
            item = self.cases.item(i); item.setHidden(query.lower() not in item.text().lower())

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Chọn thư mục MRI', str(ROOT / 'data'))
        if folder: self.populate(folder)

    def open_file(self):
        if self.loading: return
        path, _ = QFileDialog.getOpenFileName(self, 'Mở MRI NIfTI', str(ROOT / 'data'), 'NIfTI (*.nii.gz *.nii)')
        if path: self.load_volume(path)

    def open_item(self, item):
        if not self.loading: self.load_volume(item.data(Qt.ItemDataRole.UserRole))

    def start_worker(self, path, callback, volume=None):
        if self.loading: return
        if self.worker is not None:
            self.worker.wait()
        self.loading = True
        self.open_button.setEnabled(False); self.mask_button.setEnabled(False); self.cases.setEnabled(False)
        self.statusBar().showMessage('Đang đọc thể tích…')
        self.setCursor(Qt.CursorShape.BusyCursor)
        for panel in self.panels: panel.canvas.spinner.start()
        self.worker = Loader(path, volume)
        self.worker.loaded.connect(callback)
        self.worker.failed.connect(self.load_error)
        self.worker.finished.connect(self.finish_loading)
        self.worker.start()

    def finish_loading(self):
        self.loading = False
        self.open_button.setEnabled(True); self.mask_button.setEnabled(self.volume is not None); self.cases.setEnabled(True)
        for panel in self.panels: panel.canvas.spinner.stop()
        self.unsetCursor()

    def load_error(self, message):
        self.statusBar().showMessage('Không đọc được dữ liệu.')
        QMessageBox.warning(self, 'Không mở được NIfTI', message)

    def load_volume(self, path):
        self.start_worker(path, self.set_volume)

    def set_volume(self, volume):
        self.volume = volume
        self.channel.blockSignals(True); self.channel.clear(); self.channel.addItems(volume.names); self.channel.setCurrentIndex(0); self.channel.blockSignals(False); self.channel.setEnabled(True)
        self.case_title.setText(volume.path.name.removesuffix('.nii.gz').removesuffix('.nii'))
        self.case_title.setToolTip(str(volume.path) + '\n' + ('Ảnh xiên: hướng hiển thị gần đúng' if volume.oblique else 'Định hướng neurological · RAS'))
        unit = volume.image.header.get_xyzt_units()[0]
        shape = ' × '.join(str(x) for x in volume.data.shape[:3])
        spacing = ' × '.join(f'{x:.2f}' for x in volume.spacing)
        self.details.setText(f'Kích thước: {shape}\nSố kênh: {volume.data.shape[3]}\nVoxel: {spacing} {unit if unit != "unknown" else "(chưa rõ đơn vị)"}\nĐịnh dạng: NIfTI')
        for panel in self.panels:
            panel.slider.blockSignals(True); panel.slider.setRange(0, volume.data.shape[panel.axis]-1); panel.slider.setValue(volume.data.shape[panel.axis]//2); panel.slider.blockSignals(False); panel.slider.setEnabled(True)
            panel.prev_button.setEnabled(True); panel.next_button.setEnabled(True)
        self.clear_mask()
        self.level.setEnabled(True); self.width_control.setEnabled(True)
        self.infer.setEnabled(volume.data.shape[3] == 4)
        self.infer.setToolTip('Chạy checkpoint đã train (100/100 epoch, fold 0) trên ảnh đang mở.' if volume.data.shape[3] == 4
                              else 'Cần đủ 4 kênh FLAIR/T1/T1Gd/T2 (định dạng MSD Task01) để chạy checkpoint này.')
        self.reset_window()
        self.statusBar().showMessage('Đã mở MRI · Nạp mask riêng để xem nhãn · Chưa chạy AI')

    def channel_changed(self, index):
        if self.volume is not None and index >= 0:
            self.reset_window()
            if self.surface_viewer: self.surface_viewer.set_anatomy(self.volume,index)

    def reset_window(self):
        if self.volume is None: return
        low, high = self.volume.window(max(0, self.channel.currentIndex()))
        self.level.blockSignals(True); self.width_control.blockSignals(True)
        self.level.setValue((low+high)/2); self.width_control.setValue(high-low)
        self.level.blockSignals(False); self.width_control.blockSignals(False)
        self.refresh()

    def slice_changed(self, axis, value):
        self.refresh(panels=[panel for panel in self.panels if panel.axis == axis])

    def refresh(self, *args, panels=None):
        if self.volume is None: return
        channel = self.channel.currentIndex()
        if channel < 0: return
        width, center = self.width_control.value(), self.level.value()
        visible_labels = tuple(i+1 for i, toggle in enumerate(self.region_toggles) if toggle.isChecked())
        for panel in (panels if panels is not None else self.panels):
            index = panel.slider.value()
            panel.counter.setText(f'{index+1} / {self.volume.data.shape[panel.axis]}')
            panel.canvas.rgb = render_slice(self.volume, panel.axis, index, channel, (center-width/2, center+width/2), self.mask if self.show_mask.isChecked() else None, self.opacity.value()/100, visible_labels)
            remaining = [i for i in range(3) if i != panel.axis]
            panel.canvas.spacing = tuple(self.volume.spacing[remaining])
            panel.canvas.update()

    def open_mask(self):
        if self.volume is None or self.loading: return
        initial = self.volume.path.parent.parent / 'labelsTr'
        path, _ = QFileDialog.getOpenFileName(self, 'Mở mask cùng ca (nhãn 0–3)', str(initial), 'NIfTI (*.nii.gz *.nii)')
        if path: self.load_mask(path)

    def run_inference(self):
        if self.volume is None or self.inference_worker is not None: return
        self.infer.setEnabled(False)
        self.infer_dots_count = 0; self._animate_infer_text(); self.infer_dots_timer.start()
        self.infer_progress.show()
        self.statusBar().showMessage('Đang chạy nnU-Net trên ảnh đang mở…')
        for panel in self.panels: panel.canvas.spinner.start()
        self.inference_worker = InferenceWorker(self.volume)
        self.inference_worker.done.connect(self.inference_done)
        self.inference_worker.failed.connect(self.inference_failed)
        self.inference_worker.finished.connect(self.inference_finished)
        self.inference_worker.start()

    def _animate_infer_text(self):
        self.infer_dots_count = (self.infer_dots_count + 1) % 4
        self.infer.setText('⏳  Đang chạy phân đoạn' + '.' * self.infer_dots_count)

    def inference_done(self, mask_path):
        self.load_mask(mask_path, is_ai_result=True)
        self.statusBar().showMessage('Đã chạy xong phân đoạn · Kết quả AI, không phải nhãn tham chiếu')

    def inference_failed(self, message):
        self.statusBar().showMessage('Chạy phân đoạn thất bại.')
        QMessageBox.warning(self, 'Không chạy được phân đoạn', message)

    def inference_finished(self):
        self.inference_worker = None
        self.infer_dots_timer.stop(); self.infer_progress.hide()
        self.infer.setEnabled(self.volume is not None and self.volume.data.shape[3] == 4)
        self.infer.setText('▶  Chạy phân đoạn')
        for panel in self.panels: panel.canvas.spinner.stop()

    def load_mask(self, path, is_ai_result=False):
        self.start_worker(path, lambda mask: self.set_mask(mask, path, is_ai_result), self.volume)

    def set_mask(self, mask, path, is_ai_result=False):
        self.mask, self.mask_path = mask, Path(path)
        self.show_mask.setEnabled(True); self.show_mask.setChecked(True); self.clear_mask_button.setEnabled(True)
        amounts = self.volume.label_volumes(mask)
        for index, (widget, toggle) in enumerate(zip(self.region_labels, self.region_toggles)):
            count = np.count_nonzero(mask == index+1)
            widget.setText(f'{amounts[index]:.2f} mL · {count:,} voxel' if amounts else f'{count:,} voxel · thiếu đơn vị để tính mL')
            toggle.blockSignals(True); toggle.setChecked(True); toggle.blockSignals(False); toggle.setEnabled(True)
        if is_ai_result:
            source = 'Kết quả AI (checkpoint 100/100 epoch)'
            note = 'Dự đoán của model đã huấn luyện xong fold 0. Chưa đánh giá đầy đủ; không dùng để đưa ra chẩn đoán.'
        else:
            source = 'Nhãn tham chiếu' if Path(path).parent.name == 'labelsTr' else 'Mask do người dùng nạp'
            note = 'Không phải kết quả AI được chạy trong viewer.'
        self.mask_source.setText(f'{source}\n{note}')
        self.statusBar().showMessage(f'Đã nạp {source.lower()}')
        self.refresh()
        self.update_3d_input()

    def clear_mask(self):
        self.mask = self.mask_path = None
        self.show_mask.setEnabled(False); self.clear_mask_button.setEnabled(False)
        self.mask_source.setText('Chưa có mask. Nạp mask cùng ca để xem lớp phủ.')
        for widget, toggle in zip(self.region_labels, self.region_toggles):
            widget.setText('Chưa có mask')
            toggle.blockSignals(True); toggle.setChecked(True); toggle.blockSignals(False); toggle.setEnabled(False)
        self.refresh()
        self.update_3d_input()

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.statusBar().showMessage('Đang đọc dữ liệu. Vui lòng đợi hoàn tất rồi đóng cửa sổ.')
            event.ignore()
        elif self.surface_viewer and self.surface_viewer.is_busy():
            self.statusBar().showMessage('Đang dựng 3D, vui lòng đợi hoàn tất rồi đóng cửa sổ.')
            event.ignore()
        else:
            if self.surface_viewer: self.surface_viewer.shutdown()
            event.accept()


def main():
    parser = argparse.ArgumentParser(description='Native MRI desktop viewer')
    parser.add_argument('--file', type=Path)
    parser.add_argument('--mask', type=Path)
    parser.add_argument('--view', choices=['slices','3d'], default='slices')
    args = parser.parse_args()
    # Use the app's Qt/Fusion theme instead of loading optional GTK theme plugins.
    # Some GNOME installations have incompatible GSettings schemas.
    if sys.platform.startswith('linux') and not os.environ.get('QT_QPA_PLATFORMTHEME'):
        os.environ['QT_QPA_PLATFORMTHEME'] = 'generic'
    # VTK's native Qt interactor embeds an X11 OpenGL window on Linux.
    # Use XWayland on GNOME/Wayland; preserve explicit offscreen/test settings.
    if sys.platform.startswith('linux') and not os.environ.get('QT_QPA_PLATFORM'):
        os.environ['QT_QPA_PLATFORM'] = 'xcb'
    app = QApplication(sys.argv[:1]); app.setStyle('Fusion'); app.setStyleSheet(STYLE)
    app.setApplicationName('NeuroSeg')
    window = Workspace(); window.show()
    if args.mask and not args.file:
        parser.error('--mask requires --file')
    if args.file:
        def after_volume():
            if window.volume is None: return
            if args.mask:
                window.load_mask(args.mask)
                if args.view == '3d':
                    window.worker.finished.connect(lambda: QTimer.singleShot(0, lambda: window.set_focus('3d')))
            elif args.view == '3d':
                window.set_focus('3d')
        window.load_volume(args.file)
        window.worker.finished.connect(lambda: QTimer.singleShot(0, after_volume))
    elif args.view == '3d':
        window.set_focus('3d')
    sys.exit(app.exec())
