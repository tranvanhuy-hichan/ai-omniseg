"""OmniSeg workspace: a PACS-style viewer with AI segmentation, editing, measurement and export."""
import argparse
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QColor, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QHeaderView, QLabel, QListWidget, QMainWindow, QMenu, QMessageBox, QProgressBar,
    QPushButton, QScrollArea, QSlider, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QToolButton,
    QVBoxLayout, QWidget,
)

from . import registry
from .core import editing, export, inference, metrics
from .core.export import strip_nifti
from .core.volume import color_table, render_slice
from .ui.dialogs import ExportMeshDialog
from .ui.theme import ASSETS, PLANE_COLORS, STYLE
from .ui.viewport import PLANE_NAMES, SliceView
from .ui.workers import FuncWorker, InferenceWorker, Loader

ROOT = Path(__file__).resolve().parents[1]
TOOL_DEFS = [('locate', 'Định vị', 'tool_locate', 'V', 'Định vị: nhấp hoặc kéo để đặt tâm cắt ở cả ba mặt phẳng'),
             ('window', 'W/L', 'tool_window', 'W', 'Độ sáng / tương phản: kéo ngang = độ rộng cửa sổ, kéo dọc = mức (chuột phải luôn làm việc này)'),
             ('pan', 'Kéo ảnh', 'tool_pan', 'P', 'Di chuyển ảnh (chuột giữa luôn làm việc này)'),
             ('zoom', 'Zoom', 'tool_zoom', 'Z', 'Kéo dọc để phóng to / thu nhỏ (Ctrl + cuộn cũng được)'),
             ('ruler', 'Thước', 'tool_ruler', 'M', 'Đo khoảng cách: kéo giữa hai điểm'),
             ('brush', 'Vẽ', 'tool_brush', 'B', 'Vẽ vào mask bằng nhãn đang chọn ( [ và ] đổi cỡ cọ )'),
             ('eraser', 'Tẩy', 'tool_eraser', 'E', 'Xoá nhãn đang chọn khỏi mask (không đụng các nhãn khác)')]
WINDOW_STEP = {'width': 0.004, 'level': 0.002}


def section(text):
    widget = QLabel(text.upper()); widget.setObjectName('section')
    return widget


def muted(text, name='muted'):
    widget = QLabel(text); widget.setObjectName(name); widget.setWordWrap(True)
    return widget


def card():
    frame = QFrame(); frame.setObjectName('card')
    layout = QVBoxLayout(frame); layout.setContentsMargins(12, 10, 12, 10); layout.setSpacing(8)
    return frame, layout


def scroll_page():
    """(scroll area, inner layout) for one sidebar tab."""
    inner = QWidget()
    layout = QVBoxLayout(inner); layout.setContentsMargins(14, 8, 14, 16); layout.setSpacing(8)
    area = QScrollArea(); area.setWidgetResizable(True); area.setWidget(inner)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    return area, layout


def clear_layout(layout):
    while layout.count():
        widget = layout.takeAt(0).widget()
        if widget is not None:
            widget.hide(); widget.setParent(None); widget.deleteLater()


def region_card(color, name, pairs):
    """One card per region: coloured name, then one 'label ... value' line per measurement."""
    box, inner = card()
    head = QHBoxLayout(); head.setSpacing(6)
    dot = QLabel('●'); dot.setStyleSheet(f'color: rgb({color[0]}, {color[1]}, {color[2]}); font-size: 15px;')
    title = QLabel(name); title.setStyleSheet('font-weight: 700;')
    head.addWidget(dot); head.addWidget(title, 1); inner.addLayout(head)
    grid = QGridLayout(); grid.setHorizontalSpacing(12); grid.setVerticalSpacing(4); grid.setColumnStretch(1, 1)
    for row, (key, value) in enumerate(pairs):
        label = QLabel(key); label.setObjectName('muted')
        text = QLabel(value); text.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(label, row, 0); grid.addWidget(text, row, 1)
    inner.addLayout(grid)
    return box


def fmt(value, spec, none='n/a'):
    return none if value is None else format(value, spec)


class Workspace(QMainWindow):
    def __init__(self, enable_3d=None):
        super().__init__()
        self.volume = self.mask = self.reference = self.heat = None
        self.mask_path, self.mask_kind, self.mask_unsaved, self.edited = None, None, False, False
        self.worker = self.inference_worker = self.analysis_worker = self.export_worker = None
        self.loading, self.analysis = False, None
        self._analysis_again, self._pending_heat, self._last_paint = False, None, None
        self.history = editing.MaskHistory()
        self.rulers = {}
        self.tool, self.layout_mode, self.previous_mode, self.active, self.main = 'locate', '2x2', '2x2', 2, 2
        self.surface_viewer = None
        self.enable_3d = (QApplication.platformName() != 'offscreen' or bool(os.environ.get('OMNISEG_FORCE_3D'))) if enable_3d is None else enable_3d
        self.tasks = registry.list_tasks()
        self.current_task = self.tasks[0] if self.tasks else None
        self.label_ids, self.label_colors, self.entries = [], None, []
        self.setWindowTitle('OmniSeg')
        self.setWindowIcon(QIcon(f'{ASSETS}/logo.svg'))
        self.setAcceptDrops(True)
        self.resize(1560, 940)
        self.setMinimumSize(1180, 700)
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        root.addWidget(self.build_toolbar())
        root.addWidget(self.build_options_bar())
        splitter = QSplitter(Qt.Orientation.Horizontal); splitter.setHandleWidth(1)
        self.splitter = splitter
        splitter.addWidget(self.build_viewers())
        splitter.addWidget(self.build_sidebar())
        splitter.setStretchFactor(0, 1); splitter.setStretchFactor(1, 0)
        splitter.setCollapsible(0, False); splitter.setCollapsible(1, False)
        splitter.setSizes([1160, 400])
        root.addWidget(splitter, 1)
        self.analysis_timer = QTimer(self); self.analysis_timer.setSingleShot(True)
        self.analysis_timer.timeout.connect(self.after_edit)
        self.set_region_rows()
        self.set_tool('locate')
        self.arrange()
        self.update_infer_state(); self.update_edit_state(); self.update_export_state()
        self.statusBar().showMessage('Sẵn sàng · Ctrl+O mở ảnh NIfTI · kéo thả file hoặc thư mục DICOM vào cửa sổ')
        for text, key, slot in [('Mở ảnh', 'Ctrl+O', self.open_file), ('Lưu mask', 'Ctrl+S', self.save_mask),
                                ('Hoàn tác', 'Ctrl+Z', self.undo), ('Làm lại', 'Ctrl+Y', self.redo),
                                ('Về lưới mặc định', 'Esc', self.escape),
                                ('Cọ nhỏ hơn', '[', lambda: self.change_brush(-1)), ('Cọ lớn hơn', ']', lambda: self.change_brush(1))]:
            action = QAction(text, self); action.setShortcut(QKeySequence(key)); action.triggered.connect(slot); self.addAction(action)
        for tool, _, _, key, _ in TOOL_DEFS:
            action = QAction(tool, self); action.setShortcut(QKeySequence(key)); action.triggered.connect(partial(self.set_tool, tool)); self.addAction(action)
        for number, mode in (('1', '1x1'), ('2', '2x2'), ('3', '1x3')):
            action = QAction(mode, self); action.setShortcut(QKeySequence(number)); action.triggered.connect(partial(self.set_layout, mode)); self.addAction(action)
        if self.enable_3d:
            QTimer.singleShot(0, self.init_3d)
        else:
            self.cell3d_label.setText('3D tắt trong chế độ này')

    # ===== construction ========================================================================
    def tool_button(self, text, icon, tip, checkable=False, width=56):
        button = QToolButton()
        button.setText(text); button.setIcon(QIcon(f'{ASSETS}/{icon}.svg')); button.setIconSize(QSize(22, 22))
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        button.setToolTip(tip); button.setCheckable(checkable); button.setFixedWidth(width); button.setFixedHeight(50)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        return button

    @staticmethod
    def separator():
        line = QFrame(); line.setObjectName('separator'); line.setFixedHeight(34)
        return line

    def build_toolbar(self):
        bar = QFrame(); bar.setObjectName('toolbar'); bar.setFixedHeight(62)
        layout = QHBoxLayout(bar); layout.setContentsMargins(12, 4, 12, 4); layout.setSpacing(2)
        logo = QLabel(); logo.setPixmap(QIcon(f'{ASSETS}/logo.svg').pixmap(26, 26))
        brand = QLabel('OmniSeg'); brand.setObjectName('brand')
        layout.addWidget(logo); layout.addSpacing(6); layout.addWidget(brand); layout.addSpacing(14)
        self.open_button = self.tool_button('Mở', 'open', 'Mở ảnh NIfTI hoặc thư mục DICOM (Ctrl+O)')
        self.open_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        open_menu = QMenu(self.open_button)
        open_menu.addAction('Ảnh NIfTI (.nii / .nii.gz)…', self.open_file)
        open_menu.addAction('Thư mục DICOM…', self.open_dicom)
        self.open_button.setMenu(open_menu)
        layout.addWidget(self.open_button); layout.addSpacing(6); layout.addWidget(self.separator()); layout.addSpacing(6)
        self.tool_buttons, group = {}, QButtonGroup(self)
        for tool, text, icon, key, tip in TOOL_DEFS:
            button = self.tool_button(text, icon, f'{tip}  [{key}]', checkable=True, width=56)
            button.clicked.connect(partial(self.set_tool, tool))
            group.addButton(button); self.tool_buttons[tool] = button; layout.addWidget(button)
        layout.addSpacing(6); layout.addWidget(self.separator()); layout.addSpacing(6)
        self.undo_button = self.tool_button('Hoàn tác', 'undo', 'Hoàn tác nét vẽ  [Ctrl+Z]', width=62); self.undo_button.clicked.connect(self.undo)
        self.redo_button = self.tool_button('Làm lại', 'redo', 'Làm lại nét vẽ  [Ctrl+Y]', width=62); self.redo_button.clicked.connect(self.redo)
        layout.addWidget(self.undo_button); layout.addWidget(self.redo_button)
        layout.addSpacing(6); layout.addWidget(self.separator()); layout.addSpacing(6)
        self.layout_button = self.tool_button('Bố cục', 'layout_2x2', 'Chọn bố cục khung xem', width=56)
        self.layout_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        layout_menu = QMenu(self.layout_button)
        self.layout_buttons, mode_group = {}, QActionGroup(self)
        for mode, text in (('2x2', 'Bốn khung 2×2   [2]'), ('1x3', 'Một khung lớn + ba khung nhỏ   [3]'), ('1x1', 'Một khung toàn màn hình   [1]')):
            action = layout_menu.addAction(QIcon(f'{ASSETS}/layout_{mode}.svg'), text)
            action.setCheckable(True); action.triggered.connect(partial(self.set_layout, mode))
            mode_group.addAction(action); self.layout_buttons[mode] = action
        layout_menu.addSeparator()
        layout_menu.addSection('Khung chính (cho bố cục 1+3 và 1×1)')
        self.main_actions, main_group = {}, QActionGroup(self)
        for key, text in ((2, 'Axial'), (1, 'Coronal'), (0, 'Sagittal'), ('3d', '3D')):
            action = layout_menu.addAction(text)
            action.setCheckable(True); action.triggered.connect(partial(self.set_main, key))
            main_group.addAction(action); self.main_actions[key] = action
        self.layout_button.setMenu(layout_menu)
        layout.addWidget(self.layout_button)
        self.layout_buttons['2x2'].setChecked(True)
        layout.addStretch()
        self.model_choice = QComboBox(); self.model_choice.setMinimumWidth(150); self.model_choice.setToolTip('Mô hình AI')
        for task in self.tasks:
            self.model_choice.addItem(task.get('display_name', task['id']), task)
        if not self.tasks:
            self.model_choice.addItem('Không có mô hình trong tasks/'); self.model_choice.setEnabled(False)
        self.model_choice.currentIndexChanged.connect(self.model_changed)
        layout.addWidget(self.model_choice); layout.addSpacing(8)
        self.infer = QPushButton(' Chạy AI'); self.infer.setObjectName('primary')
        self.infer.setIcon(QIcon(f'{ASSETS}/run.svg')); self.infer.setIconSize(QSize(16, 16)); self.infer.setEnabled(False)
        self.infer.clicked.connect(self.infer_clicked)
        layout.addWidget(self.infer); layout.addSpacing(8)
        self.export_button = self.tool_button('Xuất', 'save', 'Lưu mask, xuất mesh 3D hoặc báo cáo')
        self.export_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.export_button)
        self.save_action = menu.addAction(QIcon(f'{ASSETS}/save.svg'), 'Lưu mask (.nii.gz)…  Ctrl+S', self.save_mask)
        self.mesh_action = menu.addAction(QIcon(f'{ASSETS}/mesh.svg'), 'Xuất mesh 3D (STL / PLY / OBJ)…', self.export_mesh)
        self.report_action = menu.addAction(QIcon(f'{ASSETS}/report.svg'), 'Xuất báo cáo (CSV / JSON)…', self.export_report)
        self.export_button.setMenu(menu)
        layout.addWidget(self.export_button)
        return bar

    def build_options_bar(self):
        """Thin second bar with the brush options; only visible for the Vẽ / Tẩy tools."""
        self.brush_box = QFrame(); self.brush_box.setObjectName('optionsbar'); self.brush_box.setFixedHeight(40)
        layout = QHBoxLayout(self.brush_box); layout.setContentsMargins(14, 4, 14, 4); layout.setSpacing(8)
        self.brush_label = QComboBox(); self.brush_label.setMinimumWidth(190)
        self.brush_label.setToolTip('Nhãn mà Vẽ ghi và Tẩy xoá (cũng chọn được bằng cách bấm vào vùng ở tab Phân đoạn)')
        self.brush_label.currentIndexChanged.connect(self.brush_label_changed)
        self.brush_size = QDoubleSpinBox(); self.brush_size.setRange(0.5, 60); self.brush_size.setValue(4.0); self.brush_size.setSingleStep(0.5)
        self.brush_size.setSuffix(' mm'); self.brush_size.setMinimumWidth(100); self.brush_size.setToolTip('Cỡ cọ  [ và ]')
        self.brush_size.valueChanged.connect(self.brush_size_changed)
        layout.addWidget(muted('Nhãn')); layout.addWidget(self.brush_label); layout.addSpacing(10)
        layout.addWidget(muted('Cỡ cọ')); layout.addWidget(self.brush_size); layout.addSpacing(14)
        hint = muted('Bấm một vùng ở tab Phân đoạn để đổi nhãn · phím [ ] đổi cỡ cọ'); hint.setWordWrap(False)
        layout.addWidget(hint)
        layout.addStretch()
        self.brush_box.hide()
        return self.brush_box

    def build_viewers(self):
        area = QWidget(); area.setStyleSheet('background: #05080b;')
        self.grid = QGridLayout(area); self.grid.setContentsMargins(3, 3, 3, 3); self.grid.setSpacing(3)
        self.views = {}
        for axis in (2, 1, 0):
            view = SliceView(axis); view.changed.connect(self.slice_changed)
            canvas = view.canvas
            canvas.activated.connect(partial(self.set_active, axis))
            canvas.expand.connect(partial(self.toggle_expand, axis))
            canvas.locate.connect(partial(self.locate, axis))
            canvas.window_drag.connect(self.window_drag)
            canvas.paint.connect(partial(self.paint, axis))
            canvas.ruler_done.connect(partial(self.ruler_done, axis))
            canvas.hovered.connect(partial(self.hovered, axis))
            self.views[axis] = view
        self.cell3d = QFrame(); self.cell3d.setObjectName('cell3d')
        self.cell3d_layout = QVBoxLayout(self.cell3d); self.cell3d_layout.setContentsMargins(1, 1, 1, 1)
        self.cell3d_label = QLabel('Đang khởi tạo 3D…'); self.cell3d_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cell3d_label.setStyleSheet('color: #5d6b7b; background: #000;')
        self.cell3d_layout.addWidget(self.cell3d_label)
        self.cells = {2: self.views[2], 1: self.views[1], 0: self.views[0], '3d': self.cell3d}
        return area

    def build_sidebar(self):
        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(370); self.tabs.setMaximumWidth(470)
        self.tabs.addTab(self.build_segment_tab(), 'Phân đoạn')
        self.tabs.addTab(self.build_display_tab(), 'Hiển thị')
        self.tabs.addTab(self.build_analysis_tab(), 'Phân tích')
        self.tab3d, self.tab3d_layout = scroll_page()
        self.tab3d_layout.addWidget(muted('Cài đặt 3D xuất hiện khi VTK sẵn sàng.'))
        self.tab3d_layout.addStretch()
        self.tabs.addTab(self.tab3d, '3D')
        return self.tabs

    def build_segment_tab(self):
        page, layout = scroll_page()
        layout.addWidget(section('Phân đoạn AI'))
        box, inner = card()
        self.model_info = muted('', 'muted')
        inner.addWidget(self.model_info)
        self.infer_reason = muted('', 'note'); self.infer_reason.hide(); inner.addWidget(self.infer_reason)
        row = QHBoxLayout(); row.addWidget(QLabel('Tính bản đồ không chắc chắn')); row.addStretch()
        self.want_uncertainty = QCheckBox(); self.want_uncertainty.setToolTip('Lưu xác suất softmax của nnU-Net và tính entropy từng voxel. Chậm hơn và tốn thêm bộ nhớ.')
        row.addWidget(self.want_uncertainty); inner.addLayout(row)
        self.infer_progress = QProgressBar(); self.infer_progress.setRange(0, 0); self.infer_progress.setTextVisible(False)
        self.infer_progress.setFixedHeight(4); self.infer_progress.hide(); inner.addWidget(self.infer_progress)
        inner.addWidget(muted('Kết quả do mô hình AI dự đoán, chưa được kiểm định lâm sàng. Không dùng để chẩn đoán.', 'note'))
        layout.addWidget(box)
        self.region_title = section('Vùng phân đoạn'); layout.addWidget(self.region_title)
        box, inner = card()
        self.region_rows = QVBoxLayout(); self.region_rows.setSpacing(6); inner.addLayout(self.region_rows)
        inner.addWidget(muted('Bấm vào một vùng để chọn nhãn cho Cọ vẽ (B) / Tẩy (E) trên thanh công cụ.'))
        self.region_labels, self.region_toggles = [], []
        layout.addWidget(box)
        box, inner = card()
        self.mask_source = muted('Chưa có mask. Chạy AI, nạp mask, hoặc dùng Cọ vẽ để tự tạo.')
        inner.addWidget(self.mask_source)
        buttons = QHBoxLayout(); buttons.setSpacing(6)
        self.mask_button = QPushButton('Nạp mask…'); self.mask_button.clicked.connect(self.open_mask)
        self.clear_mask_button = QPushButton('Bỏ mask'); self.clear_mask_button.clicked.connect(self.clear_mask)
        self.save_button = QPushButton('Lưu mask…'); self.save_button.clicked.connect(self.save_mask)
        for button in (self.mask_button, self.clear_mask_button, self.save_button):
            buttons.addWidget(button)
        inner.addLayout(buttons)
        layout.addWidget(box)
        layout.addWidget(section('Lớp phủ'))
        box, inner = card()
        row = QHBoxLayout(); row.addWidget(QLabel('Hiện mask')); row.addStretch()
        self.show_mask = QCheckBox(); self.show_mask.setChecked(True); self.show_mask.toggled.connect(self.refresh)
        row.addWidget(self.show_mask); inner.addLayout(row)
        row = QHBoxLayout(); row.addWidget(muted('Độ phủ'))
        self.opacity = QSlider(Qt.Orientation.Horizontal); self.opacity.setRange(0, 100); self.opacity.setValue(40)
        self.opacity.valueChanged.connect(self.refresh); row.addWidget(self.opacity, 1); inner.addLayout(row)
        row = QHBoxLayout(); row.addWidget(QLabel('Bản đồ không chắc chắn')); row.addStretch()
        self.show_heat = QCheckBox(); self.show_heat.setEnabled(False); self.show_heat.toggled.connect(self.refresh)
        self.show_heat.setToolTip('Entropy softmax của AI: sáng = mô hình kém chắc chắn. Chỉ có khi bật "Tính bản đồ không chắc chắn" lúc chạy.')
        row.addWidget(self.show_heat); inner.addLayout(row)
        layout.addWidget(box)
        layout.addStretch()
        return page

    def build_display_tab(self):
        page, layout = scroll_page()
        layout.addWidget(section('Kênh ảnh'))
        self.channel = QComboBox(); self.channel.setEnabled(False); self.channel.currentIndexChanged.connect(self.channel_changed)
        layout.addWidget(self.channel)
        layout.addWidget(section('Cửa sổ hiển thị (Window / Level)'))
        box, inner = card()
        grid = QGridLayout(); grid.setHorizontalSpacing(10); grid.setVerticalSpacing(3)
        self.level, self.width_control = QDoubleSpinBox(), QDoubleSpinBox()
        for control in (self.level, self.width_control):
            control.setRange(-1e6, 1e6); control.setDecimals(1); control.setEnabled(False); control.valueChanged.connect(self.refresh)
        self.width_control.setMinimum(0.1)
        grid.addWidget(muted('Level (mức)'), 0, 0); grid.addWidget(muted('Width (độ rộng)'), 0, 1)
        grid.addWidget(self.level, 1, 0); grid.addWidget(self.width_control, 1, 1)
        inner.addLayout(grid)
        self.preset_row = QWidget(); preset_layout = QHBoxLayout(self.preset_row); preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_layout.addWidget(muted('Cài sẵn CT'))
        self.preset = QComboBox(); self.preset.addItem('Chọn…', None)
        for name, level, width in metrics.CT_PRESETS:
            self.preset.addItem(f'{name}  ({level:+d} / {width})', (level, width))
        self.preset.activated.connect(self.apply_preset); preset_layout.addWidget(self.preset, 1)
        inner.addWidget(self.preset_row); self.preset_row.hide()
        buttons = QHBoxLayout()
        reset = QPushButton('Đặt lại W/L'); reset.clicked.connect(self.reset_window)
        reset_view = QPushButton('Đặt lại phóng to'); reset_view.clicked.connect(self.reset_views)
        buttons.addWidget(reset); buttons.addWidget(reset_view); inner.addLayout(buttons)
        layout.addWidget(box)
        layout.addWidget(section('Khung nhìn'))
        box, inner = card()
        row = QHBoxLayout(); row.addWidget(QLabel('Đường định vị')); row.addStretch()
        self.show_cross = QCheckBox(); self.show_cross.setChecked(True); self.show_cross.toggled.connect(self.refresh)
        row.addWidget(self.show_cross); inner.addLayout(row)
        inner.addWidget(muted('Cuộn chuột: đổi lát · Ctrl + cuộn: phóng to · Chuột phải kéo: W/L · Chuột giữa kéo: di chuyển · Nhấp đúp: phóng khung', 'muted'))
        layout.addWidget(box)
        layout.addWidget(section('Thông tin ảnh'))
        box, inner = card()
        self.details = muted('Chưa mở ảnh.'); self.details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        inner.addWidget(self.details); layout.addWidget(box)
        layout.addStretch()
        return page

    def build_analysis_tab(self):
        page, layout = scroll_page()
        layout.addWidget(section('Số đo từng vùng'))
        self.stats_box = QVBoxLayout(); self.stats_box.setSpacing(8)
        layout.addLayout(self.stats_box)
        self.stats_note = muted('Ø lớn nhất tính cho khối liên thông lớn nhất của vùng. Cường độ lấy từ kênh đang xem.')
        layout.addWidget(self.stats_note)
        layout.addWidget(section('So sánh với mask tham chiếu'))
        box, inner = card()
        buttons = QHBoxLayout(); buttons.setSpacing(6)
        self.ref_button = QPushButton('Nạp mask tham chiếu…'); self.ref_button.clicked.connect(self.open_reference)
        self.ref_clear = QPushButton('Bỏ'); self.ref_clear.clicked.connect(self.clear_reference)
        buttons.addWidget(self.ref_button, 1); buttons.addWidget(self.ref_clear); inner.addLayout(buttons)
        row = QHBoxLayout(); row.addWidget(QLabel('Hiện viền tham chiếu (trắng)')); row.addStretch()
        self.show_ref = QCheckBox(); self.show_ref.setChecked(True); self.show_ref.toggled.connect(self.refresh); row.addWidget(self.show_ref)
        inner.addLayout(row)
        self.ref_info = muted('Nạp ground truth để tính Dice / IoU với mask hiện tại.')
        inner.addWidget(self.ref_info)
        layout.addWidget(box)
        self.compare_box = QVBoxLayout(); self.compare_box.setSpacing(8)
        layout.addLayout(self.compare_box)
        layout.addWidget(section('Phép đo khoảng cách'))
        self.ruler_list = QListWidget(); self.ruler_list.setMaximumHeight(130); layout.addWidget(self.ruler_list)
        clear = QPushButton('Xoá các phép đo'); clear.clicked.connect(self.clear_rulers); layout.addWidget(clear)
        layout.addStretch()
        self.set_analysis_empty()
        return page

    # ===== 3D ==================================================================================
    def init_3d(self):
        if self.surface_viewer is not None:
            return
        try:
            from .view3d import SurfaceViewer
            self.surface_viewer = SurfaceViewer(self.entries, registry.label_opacities(self.current_task) if self.current_task else None)
        except Exception as error:
            self.cell3d_label.setText(f'Không có 3D ({error}).\nCài VTK: make setup-desktop')
            return
        self.cell3d_layout.removeWidget(self.cell3d_label); self.cell3d_label.hide()
        self.cell3d_layout.addWidget(self.surface_viewer)
        while self.tab3d_layout.count():
            item = self.tab3d_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.tab3d_layout.addWidget(self.surface_viewer.settings); self.tab3d_layout.addStretch()
        self.surface_viewer.vtk.installEventFilter(self)
        self.surface_viewer.activate()
        self.update_3d_input()

    def eventFilter(self, obj, event):
        if self.surface_viewer is not None and obj is self.surface_viewer.vtk:
            if event.type() == QEvent.Type.MouseButtonPress:
                self.set_active('3d')
            elif event.type() == QEvent.Type.MouseButtonDblClick:
                self.toggle_expand('3d')
        return False

    def update_3d_input(self):
        if self.surface_viewer is None:
            return
        self.surface_viewer.set_anatomy(self.volume, max(0, self.channel.currentIndex()), (self.current_task or {}).get('modality'))
        if self.volume is None or self.mask is None:
            self.surface_viewer.set_input(None)
        else:
            self.surface_viewer.set_input(self.mask, self.volume.image.affine, self.mask_source.text())

    # ===== layout and tools ====================================================================
    def arrange(self):
        for widget in self.cells.values():
            self.grid.removeWidget(widget)
        for i in range(4):
            self.grid.setRowStretch(i, 0); self.grid.setColumnStretch(i, 0)
        mode, active = self.layout_mode, self.main
        self.main_actions[active].setChecked(True)
        for action in self.main_actions.values():
            action.setEnabled(mode != '2x2')
        self.layout_button.setIcon(QIcon(f'{ASSETS}/layout_{mode}.svg'))
        if mode == '1x1':
            for key, widget in self.cells.items():
                widget.setVisible(key == active)
            self.grid.addWidget(self.cells[active], 0, 0)
            self.grid.setRowStretch(0, 1); self.grid.setColumnStretch(0, 1)
        elif mode == '2x2':
            for key, (row, column) in {2: (0, 0), 1: (0, 1), 0: (1, 0), '3d': (1, 1)}.items():
                self.grid.addWidget(self.cells[key], row, column); self.cells[key].show()
            for i in range(2):
                self.grid.setRowStretch(i, 1); self.grid.setColumnStretch(i, 1)
        else:
            self.grid.addWidget(self.cells[active], 0, 0, 3, 1); self.cells[active].show()
            for row, key in enumerate(k for k in self.cells if k != active):
                self.grid.addWidget(self.cells[key], row, 1); self.cells[key].show()
                self.grid.setRowStretch(row, 1)
            self.grid.setColumnStretch(0, 3); self.grid.setColumnStretch(1, 1)
        self.layout_buttons[mode].setChecked(True)
        self.set_active(active)
        if self.surface_viewer is not None:
            QTimer.singleShot(0, self.surface_viewer.render)

    def set_layout(self, mode):
        if mode == '1x1' and self.layout_mode != '1x1':
            self.previous_mode = self.layout_mode
        elif mode != '1x1':
            self.previous_mode = mode
        self.layout_mode = mode
        self.arrange()

    def set_main(self, key):
        self.main = self.active = key
        self.arrange()

    def toggle_expand(self, key):
        if self.layout_mode == '1x3' and key != self.main:
            self.set_main(key)          # promote a small viewport to the large slot
            return
        self.main = self.active = key
        if self.layout_mode == '1x1':
            self.layout_mode = self.previous_mode if self.previous_mode != '1x1' else '2x2'
        else:
            self.previous_mode, self.layout_mode = self.layout_mode, '1x1'
        self.arrange()

    def escape(self):
        if self.layout_mode == '1x1':
            self.set_layout(self.previous_mode if self.previous_mode != '1x1' else '2x2')

    def set_active(self, key):
        self.active = key
        for axis, view in self.views.items():
            view.canvas.active = axis == key; view.canvas.update()
        color = PLANE_COLORS['3d'] if key == '3d' else '#2a3441'
        self.cell3d.setStyleSheet(f'QFrame#cell3d {{ background: #000; border: {2 if key == "3d" else 1}px solid {color}; }}')

    def set_tool(self, tool):
        self.tool = tool
        self.tool_buttons[tool].setChecked(True)
        for view in self.views.values():
            view.canvas.set_tool(tool)
        hints = {'locate': 'Định vị: nhấp hoặc kéo trong ảnh', 'window': 'W/L: kéo chuột để đổi độ sáng/tương phản',
                 'pan': 'Kéo để di chuyển ảnh', 'zoom': 'Kéo lên/xuống để phóng to/thu nhỏ', 'ruler': 'Kéo giữa hai điểm để đo khoảng cách',
                 'brush': 'Cọ vẽ: kéo để vẽ nhãn đang chọn', 'eraser': 'Tẩy: kéo để xoá nhãn đang chọn'}
        self.statusBar().showMessage(hints[tool])
        self.brush_box.setVisible(tool in ('brush', 'eraser'))
        self.brush_label_changed()

    # ===== regions / model =====================================================================
    def set_region_rows(self):
        """(Re)build region rows, colours and 3D switches from the selected task's labels."""
        task = self.current_task
        self.entries = registry.label_entries(task) if task else []
        while self.region_rows.count():
            widget = self.region_rows.takeAt(0).widget()
            if widget is not None:
                widget.hide(); widget.setParent(None); widget.deleteLater()
        self.region_labels, self.region_toggles, self.region_widgets = [], [], []
        self.label_ids = [label_id for label_id, _, _ in self.entries]
        self.label_colors = color_table(self.entries)
        name = task.get('display_name', task['id']) if task else 'chưa có mô hình'
        self.region_title.setText(f'VÙNG PHÂN ĐOẠN · {name.upper()}')
        self.brush_label.blockSignals(True); self.brush_label.clear()
        for position, (label_id, text, color) in enumerate(self.entries):
            self.brush_label.addItem(f'{label_id} · {text}', label_id)
            row_widget = QWidget(); row_widget.setObjectName('regionRow'); row_widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
            row_widget.setCursor(Qt.CursorShape.PointingHandCursor)
            row_widget.setToolTip('Bấm để chọn vùng này làm nhãn cho Cọ vẽ / Tẩy')
            row_widget.mousePressEvent = lambda event, i=position: self.brush_label.setCurrentIndex(i)
            self.region_widgets.append(row_widget)
            row_layout = QHBoxLayout(row_widget); row_layout.setContentsMargins(0, 0, 0, 0); row_layout.setSpacing(0)
            dot = QLabel('●'); dot.setStyleSheet(f'color: rgb({color[0]}, {color[1]}, {color[2]}); font-size: 16px;'); dot.setFixedWidth(18)
            text_widget = QWidget(); text_layout = QVBoxLayout(text_widget); text_layout.setContentsMargins(6, 0, 0, 0); text_layout.setSpacing(1)
            name_label = QLabel(text); name_label.setStyleSheet('font-weight: 600;')
            amount = muted('Chưa có mask')
            text_layout.addWidget(name_label); text_layout.addWidget(amount)
            toggle = QCheckBox(); toggle.setChecked(True); toggle.setEnabled(False); toggle.toggled.connect(self.refresh)
            row_layout.addWidget(dot, 0, Qt.AlignmentFlag.AlignTop); row_layout.addWidget(text_widget, 1)
            row_layout.addWidget(toggle, 0, Qt.AlignmentFlag.AlignVCenter)
            self.region_labels.append(amount); self.region_toggles.append(toggle)
            self.region_rows.addWidget(row_widget)
        self.brush_label.blockSignals(False)
        self.brush_label_changed()
        if task:
            channels = ', '.join(task['channel_names'].values())
            self.model_info.setText(f"{task.get('modality', '')} · đầu vào: {channels} · cấu hình {task.get('nnunet_config', '')}")
        if self.surface_viewer is not None:
            self.surface_viewer.set_labels(self.entries, registry.label_opacities(task) if task else None)

    def infer_block_reason(self):
        task = self.current_task
        if task is None:
            return 'Chưa có mô hình nào để chạy.'
        if not inference.checkpoint_exists(task):
            return 'Chưa có checkpoint cho model này. Cần train xong và đặt checkpoint vào artifacts/nnUNet_results.'
        if self.volume is not None and self.volume.data.shape[3] != inference.channel_count(task):
            return f'Cần đủ {inference.channel_count(task)} kênh ảnh đầu vào để chạy mô hình đang chọn.'
        return None

    def update_infer_state(self):
        reason = self.infer_block_reason()
        running = self.inference_worker is not None
        self.infer.setEnabled(running or (reason is None and self.volume is not None))
        self.infer_reason.setText(reason or ''); self.infer_reason.setVisible(bool(reason))
        self.infer.setToolTip('Bấm để huỷ' if running else reason or ('Chạy checkpoint đã train cho mô hình đang chọn trên ảnh đang mở.'
                                                                      if self.volume is not None else 'Mở ảnh để chạy mô hình.'))

    def model_changed(self, index):
        task = self.model_choice.itemData(index)
        if task is None:
            return
        self.current_task = task
        self.set_region_rows()
        if self.mask is not None or self.reference is not None:
            self.clear_mask(); self.clear_reference()
            self.statusBar().showMessage('Đã đổi mô hình · bỏ mask cũ vì mỗi mô hình có bộ nhãn riêng')
        self.update_infer_state()
        self.update_preset_row()

    # ===== loading =============================================================================
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls and not self.loading:
            self.load_volume(urls[0].toLocalFile())

    def open_file(self):
        if self.loading:
            return
        path, _ = QFileDialog.getOpenFileName(self, 'Mở ảnh NIfTI', str(ROOT / 'data'), 'NIfTI (*.nii.gz *.nii)')
        if path:
            self.load_volume(path)

    def open_dicom(self):
        if self.loading:
            return
        folder = QFileDialog.getExistingDirectory(self, 'Chọn thư mục chứa chuỗi DICOM')
        if folder:
            self.load_volume(folder)

    def start_worker(self, path, callback, volume=None):
        if self.loading:
            return
        if self.worker is not None:
            self.worker.wait()
        self.loading = True
        self.open_button.setEnabled(False)
        self.statusBar().showMessage('Đang đọc dữ liệu…')
        self.setCursor(Qt.CursorShape.BusyCursor)
        for view in self.views.values():
            view.canvas.spinner.start()
        self.worker = Loader(path, volume, tuple(self.label_ids))
        self.worker.loaded.connect(callback)
        self.worker.failed.connect(self.load_error)
        self.worker.finished.connect(self.finish_loading)
        self.worker.start()

    def finish_loading(self):
        self.loading = False
        self.open_button.setEnabled(True)
        for view in self.views.values():
            view.canvas.spinner.stop()
        self.unsetCursor()
        self.update_export_state()

    def load_error(self, message):
        self.statusBar().showMessage('Không đọc được dữ liệu.')
        QMessageBox.warning(self, 'Không mở được dữ liệu', message)

    def load_volume(self, path):
        self.start_worker(path, self.set_volume)

    def set_volume(self, volume):
        self.volume = volume
        name = strip_nifti(volume.path.name)
        self.channel.blockSignals(True); self.channel.clear(); self.channel.addItems(volume.names)
        self.channel.setCurrentIndex(0); self.channel.blockSignals(False); self.channel.setEnabled(True)
        self.setWindowTitle(f'OmniSeg · {name}')
        unit = volume.image.header.get_xyzt_units()[0]
        shape = ' × '.join(str(x) for x in volume.data.shape[:3])
        spacing = ' × '.join(f'{x:.2f}' for x in volume.spacing)
        self.details.setText(f'Tệp: {volume.path.name}\nKích thước: {shape}\nSố kênh: {volume.data.shape[3]}\n'
                             f'Voxel: {spacing} {unit if unit != "unknown" else "(chưa rõ đơn vị)"}\n'
                             f'Hướng: {"xiên (hiển thị gần đúng)" if volume.oblique else "RAS (đã chuẩn hoá)"}')
        self.rulers.clear(); self.ruler_list.clear()
        for axis, view in self.views.items():
            view.slider.blockSignals(True); view.slider.setRange(0, volume.data.shape[axis] - 1)
            view.slider.setValue(volume.data.shape[axis] // 2); view.slider.blockSignals(False); view.slider.setEnabled(True)
            view.canvas.reset_view()
        self.suggest_model()
        self.clear_mask(); self.clear_reference()
        self.level.setEnabled(True); self.width_control.setEnabled(True)
        self.update_preset_row()
        self.update_infer_state()
        self.reset_window()
        self.update_3d_input()
        self.update_export_state()
        self.statusBar().showMessage('Đã mở ảnh · chọn mô hình rồi bấm "Chạy phân đoạn AI", hoặc nạp mask có sẵn')

    def suggest_model(self):
        """Select the task whose input channels (and CT/MRI modality) match the image just opened."""
        channels = self.volume.data.shape[3]
        matches = [(i, t) for i, t in enumerate(self.tasks) if inference.channel_count(t) == channels]
        if len(matches) > 1:
            ct = self.volume.is_ct()
            matches = [m for m in matches if (str(m[1].get('modality', '')).upper() == 'CT') == ct] or matches
        if matches and self.model_choice.currentIndex() != matches[0][0]:
            self.model_choice.blockSignals(True); self.model_choice.setCurrentIndex(matches[0][0]); self.model_choice.blockSignals(False)
            self.current_task = matches[0][1]
            self.set_region_rows()
            self.statusBar().showMessage(f'Đã chọn mô hình phù hợp với ảnh: {self.current_task.get("display_name")}')

    def update_preset_row(self):
        self.preset_row.setVisible(self.volume is not None and self.volume.is_ct((self.current_task or {}).get('modality')))

    def apply_preset(self, index):
        data = self.preset.itemData(index)
        if data is None:
            return
        level, width = data
        for control, value in ((self.level, level), (self.width_control, width)):
            control.blockSignals(True); control.setValue(value); control.blockSignals(False)
        self.refresh()

    def channel_changed(self, index):
        if self.volume is not None and index >= 0:
            self.reset_window()
            if self.surface_viewer:
                self.surface_viewer.set_anatomy(self.volume, index, (self.current_task or {}).get('modality'))
            self.schedule_analysis()

    # ===== slice display =======================================================================
    def reset_window(self):
        if self.volume is None:
            return
        low, high = self.volume.window(max(0, self.channel.currentIndex()))
        if self.is_ct():
            low, high = metrics.CT_PRESETS[0][1] - metrics.CT_PRESETS[0][2] / 2, metrics.CT_PRESETS[0][1] + metrics.CT_PRESETS[0][2] / 2
        for control, value in ((self.level, (low + high) / 2), (self.width_control, high - low)):
            control.blockSignals(True); control.setValue(value); control.blockSignals(False)
        self.preset.setCurrentIndex(0)
        self.refresh()

    def reset_views(self):
        for view in self.views.values():
            view.canvas.reset_view()
        self.refresh()

    def window_drag(self, dx, dy):
        if self.volume is None:
            return
        width = max(0.1, self.width_control.value() * (1 + dx * WINDOW_STEP['width']))
        level = self.level.value() + dy * max(self.width_control.value(), 1) * WINDOW_STEP['level']
        for control, value in ((self.width_control, width), (self.level, level)):
            control.blockSignals(True); control.setValue(value); control.blockSignals(False)
        self.refresh()

    def slice_changed(self, axis, value):
        self.refresh()

    def locate(self, axis, u, v):
        if self.volume is None:
            return
        remaining = [i for i in range(3) if i != axis]
        for dim, value in zip(remaining, (u, v)):
            slider = self.views[dim].slider
            slider.blockSignals(True); slider.setValue(value); slider.blockSignals(False)
        self.refresh()

    def is_ct(self):
        return self.volume is not None and self.volume.is_ct((self.current_task or {}).get('modality'))

    def refresh(self, *args):
        if self.volume is None:
            for view in self.views.values():
                view.canvas.rgb = None; view.canvas.update()
            return
        channel = self.channel.currentIndex()
        if channel < 0:
            return
        width, level = self.width_control.value(), self.level.value()
        visible = tuple(label_id for label_id, toggle in zip(self.label_ids, self.region_toggles) if toggle.isChecked())
        mask = self.mask if self.mask is not None and self.show_mask.isChecked() else None
        reference = self.reference if self.reference is not None and self.show_ref.isChecked() else None
        heat = self.heat if self.heat is not None and self.show_heat.isChecked() else None
        name = strip_nifti(self.volume.path.name)
        banner = self.banner_text()
        for axis, view in self.views.items():
            canvas, index = view.canvas, view.slider.value()
            remaining = [i for i in range(3) if i != axis]
            canvas.rgb = render_slice(self.volume, axis, index, channel, (level - width / 2, level + width / 2), mask,
                                      self.opacity.value() / 100, visible, self.label_colors, reference, heat)
            canvas.spacing = tuple(float(x) for x in self.volume.spacing[remaining])
            canvas.plane_shape = tuple(self.volume.data.shape[i] for i in remaining)
            canvas.cursor_voxel = tuple(self.views[i].slider.value() for i in remaining)
            canvas.show_cross = self.show_cross.isChecked()
            canvas.rulers = self.rulers.get((axis, index), [])
            canvas.banner = banner
            canvas.corners = {
                'tl': [name, self.volume.names[channel]],
                'tr': [f'W {width:.0f}   L {level:.0f}', f'Zoom {canvas.zoom * 100:.0f}%'],
                'bl': [f'Lát {index + 1} / {self.volume.data.shape[axis]}', f'Độ dày {self.volume.spacing[axis]:.2f} mm'],
                'br': [],
            }
            canvas.update()

    def hovered(self, axis, voxel):
        canvas = self.views[axis].canvas
        text = ''
        if voxel is not None and self.volume is not None:
            remaining = [i for i in range(3) if i != axis]
            index = [0, 0, 0]
            index[axis] = self.views[axis].slider.value(); index[remaining[0]], index[remaining[1]] = voxel
            value = self.volume.voxel_value(max(0, self.channel.currentIndex()), index)
            if value is not None:
                text = f'({index[0]}, {index[1]}, {index[2]})  {value:.1f}{" HU" if self.is_ct() else ""}'
                if self.mask is not None:
                    label_id = int(self.mask[tuple(index)])
                    names = {lid: name for lid, name, _ in self.entries}
                    if label_id:
                        text += f'  ·  {names.get(label_id, label_id)}'
        if text != canvas.probe:
            canvas.probe = text; canvas.update()

    def banner_text(self):
        if self.mask is None:
            return ''
        if self.edited:
            return 'MASK ĐÃ CHỈNH SỬA THỦ CÔNG'
        return ''

    # ===== measuring ===========================================================================
    def ruler_done(self, axis, start, end):
        index = self.views[axis].slider.value()
        self.rulers.setdefault((axis, index), []).append((start, end))
        spacing = [self.volume.spacing[i] for i in range(3) if i != axis]
        mm = float(np.hypot((end[0] - start[0]) * spacing[0], (end[1] - start[1]) * spacing[1]))
        self.ruler_list.addItem(f'{PLANE_NAMES[axis].capitalize()} · lát {index + 1}: {mm:.1f} mm')
        self.refresh()

    def clear_rulers(self):
        self.rulers.clear(); self.ruler_list.clear(); self.refresh()

    # ===== editing =============================================================================
    def brush_label_changed(self, *args):
        label_id = self.brush_label.currentData()
        color = next((c for lid, _, c in self.entries if lid == label_id), (255, 255, 255))
        for view in self.views.values():
            view.canvas.brush_color = QColor(*color); view.canvas.update()
        for position, widget in enumerate(getattr(self, 'region_widgets', [])):
            selected = position == self.brush_label.currentIndex() and self.tool in ('brush', 'eraser')
            widget.setStyleSheet('#regionRow { background: #1f3a44; border-radius: 6px; }' if selected else '')

    def brush_size_changed(self, value):
        for view in self.views.values():
            view.canvas.brush_mm = value; view.canvas.update()

    def change_brush(self, direction):
        self.brush_size.setValue(max(0.5, self.brush_size.value() + direction * (1.0 if self.brush_size.value() < 10 else 2.0)))

    def ensure_mask(self):
        if self.volume is None or self.brush_label.currentData() is None:
            return False
        if self.mask is None:
            self.set_mask(np.zeros(self.volume.data.shape[:3], np.uint8), None, kind='drawn')
        return True

    def paint(self, axis, u, v, phase):
        if phase == 'end':
            if self.mask is not None and self.history.commit(self.mask):
                self.edited = True; self.mask_unsaved = True
                self.update_edit_state(); self.refresh()
                self.analysis_timer.start(600)
            self._last_paint = None
            return
        if not self.ensure_mask():
            return
        label_id = self.brush_label.currentData()
        erase = self.tool == 'eraser'
        index = self.views[axis].slider.value()
        if phase == 'start':
            self.history.begin(); self._last_paint = None
        points = [(u, v)]
        if self._last_paint is not None:
            steps = max(abs(u - self._last_paint[0]), abs(v - self._last_paint[1]))
            if steps > 1:
                points = [(round(self._last_paint[0] + (u - self._last_paint[0]) * t / steps),
                           round(self._last_paint[1] + (v - self._last_paint[1]) * t / steps)) for t in range(1, steps + 1)]
        self._last_paint = (u, v)
        for point in points:
            self.history.edit(self.mask, axis, index, partial(
                editing.apply_brush, self.mask, axis, index, point, self.brush_size.value(), self.volume.spacing,
                0 if erase else label_id, label_id if erase else None))
        self.refresh()

    def undo(self):
        if self.mask is not None and self.history.undo(self.mask):
            self.mask_unsaved = True; self.after_history_change()

    def redo(self):
        if self.mask is not None and self.history.redo(self.mask):
            self.mask_unsaved = True; self.after_history_change()

    def after_history_change(self):
        self.edited = True
        self.update_edit_state(); self.refresh(); self.analysis_timer.start(300)

    def after_edit(self):
        """Debounced: refresh amounts, measurements and the 3D surfaces after the mask changed."""
        self.update_region_amounts()
        self.run_analysis()
        self.update_3d_input()
        self.update_export_state()

    def update_edit_state(self):
        self.undo_button.setEnabled(self.history.can_undo)
        self.redo_button.setEnabled(self.history.can_redo)

    # ===== masks ===============================================================================
    def open_mask(self):
        if self.volume is None or self.loading:
            return
        initial = self.volume.path.parent.parent / 'labelsTr'
        label_ids = (self.current_task or {}).get('labels', {}).values()
        label_range = f'nhãn 0–{max(label_ids)}' if label_ids else 'nhãn'
        path, _ = QFileDialog.getOpenFileName(self, f'Mở mask cùng ca ({label_range})', str(initial if initial.is_dir() else self.volume.path.parent), 'NIfTI (*.nii.gz *.nii)')
        if path:
            self.load_mask(path)

    def load_mask(self, path, is_ai_result=False):
        self.start_worker(path, lambda mask: self.set_mask(mask, path, is_ai_result), self.volume)

    def update_region_amounts(self):
        if self.mask is None:
            for widget, toggle in zip(self.region_labels, self.region_toggles):
                widget.setText('Chưa có mask')
                toggle.blockSignals(True); toggle.setChecked(True); toggle.blockSignals(False); toggle.setEnabled(False)
            return
        amounts = self.volume.label_volumes(self.mask, tuple(self.label_ids))
        for index, (label_id, widget, toggle) in enumerate(zip(self.label_ids, self.region_labels, self.region_toggles)):
            count = int(np.count_nonzero(self.mask == label_id))
            widget.setText(f'{amounts[index]:.2f} mL · {count:,} voxel' if amounts else f'{count:,} voxel · thiếu đơn vị để tính mL')
            toggle.setEnabled(True)

    def set_mask(self, mask, path, is_ai_result=False, kind=None):
        kind = kind or ('ai' if is_ai_result else ('reference' if path and Path(path).parent.name == 'labelsTr' else 'user'))
        self.mask, self.mask_path, self.mask_kind = mask, (Path(path) if path else None), kind
        self.mask_unsaved, self.edited = kind in ('ai', 'drawn'), False
        self.history.clear()
        self.heat, self._pending_heat = (self._pending_heat if kind == 'ai' else None), None
        self.show_heat.setEnabled(self.heat is not None)
        self.show_heat.blockSignals(True); self.show_heat.setChecked(self.heat is not None); self.show_heat.blockSignals(False)
        for toggle in self.region_toggles:
            toggle.blockSignals(True); toggle.setChecked(True); toggle.blockSignals(False)
        self.show_mask.setChecked(True)
        self.update_region_amounts()
        model_name = self.current_task.get('display_name', self.current_task['id']) if self.current_task else 'mô hình'
        source, note = {
            'ai': (f'Kết quả AI ({model_name})', 'Dự đoán của model đã huấn luyện. Chưa đánh giá đầy đủ; không dùng để chẩn đoán. Hãy lưu mask nếu muốn giữ lại.'),
            'reference': ('Nhãn tham chiếu', 'Không phải kết quả AI được chạy trong viewer.'),
            'drawn': ('Mask tự vẽ', 'Tạo bằng Cọ vẽ. Nhớ lưu mask.'),
        }.get(kind, ('Mask do người dùng nạp', 'Không phải kết quả AI được chạy trong viewer.'))
        self.mask_source.setText(f'{source}\n{note}')
        self.statusBar().showMessage(f'Đã nạp {source.lower()}')
        self.refresh()
        self.update_3d_input()
        self.update_edit_state(); self.update_export_state()
        self.schedule_analysis(0)

    def clear_mask(self):
        self.mask = self.mask_path = self.mask_kind = self.heat = None
        self.mask_unsaved = self.edited = False
        self.history.clear()
        self.show_heat.setEnabled(False)
        self.mask_source.setText('Chưa có mask. Chạy AI, nạp mask, hoặc dùng Cọ vẽ để tự tạo.')
        self.update_region_amounts()
        self.refresh(); self.update_3d_input(); self.update_edit_state(); self.update_export_state()
        self.schedule_analysis(0)

    def open_reference(self):
        if self.volume is None or self.loading:
            return
        initial = self.volume.path.parent.parent / 'labelsTr'
        path, _ = QFileDialog.getOpenFileName(self, 'Mở mask tham chiếu (ground truth)', str(initial if initial.is_dir() else self.volume.path.parent), 'NIfTI (*.nii.gz *.nii)')
        if path:
            self.start_worker(path, lambda mask: self.set_reference(mask, path), self.volume)

    def set_reference(self, mask, path):
        self.reference = mask
        self.ref_info.setText(f'Tham chiếu: {Path(path).name}')
        self.refresh(); self.schedule_analysis(0)

    def clear_reference(self):
        self.reference = None
        self.ref_info.setText('Nạp ground truth để tính Dice / IoU với mask hiện tại.')
        self.refresh(); self.schedule_analysis(0)

    def save_mask(self):
        if self.volume is None or self.mask is None:
            self.statusBar().showMessage('Chưa có mask để lưu.')
            return None
        default = self.volume.path.parent / f'{strip_nifti(self.volume.path.name)}_mask.nii.gz'
        path, _ = QFileDialog.getSaveFileName(self, 'Lưu mask', str(default), 'NIfTI (*.nii.gz *.nii)')
        if not path:
            return None
        return self.write_mask(path)

    def write_mask(self, path):
        try:
            saved = export.save_mask(self.volume, self.mask, path)
        except Exception as error:
            QMessageBox.warning(self, 'Không lưu được mask', str(error))
            return None
        self.mask_unsaved = False; self.mask_path = saved
        self.update_edit_state()
        self.statusBar().showMessage(f'Đã lưu mask: {saved} (cùng hướng và affine với ảnh gốc)')
        return saved

    # ===== analysis ============================================================================
    def set_analysis_empty(self):
        self.stats_rows, self.compare_rows = [], []
        clear_layout(self.stats_box); clear_layout(self.compare_box)
        self.stats_box.addWidget(muted('Chưa có mask để đo.'))

    def schedule_analysis(self, delay=0):
        self.analysis_timer.stop()
        QTimer.singleShot(delay, self.run_analysis)

    def run_analysis(self):
        if self.volume is None or self.mask is None:
            self.analysis = None; self.set_analysis_empty()
            return
        if self.analysis_worker is not None:
            self._analysis_again = True
            return
        channel = max(0, self.channel.currentIndex())
        worker = FuncWorker(metrics.analyse, self.volume, channel, self.mask.copy(), self.reference, tuple(self.label_ids))
        worker.result.connect(self.analysis_done)
        worker.failed.connect(lambda message: self.statusBar().showMessage(f'Không phân tích được: {message}'))
        worker.finished.connect(self.analysis_finished)
        self.analysis_worker = worker
        worker.start()

    def analysis_finished(self):
        self.analysis_worker = None
        if self._analysis_again:
            self._analysis_again = False
            self.run_analysis()

    def analysis_done(self, result):
        if self.mask is None:
            return
        self.analysis = result
        names = {label_id: name for label_id, name, _ in self.entries}
        colors = {label_id: color for label_id, _, color in self.entries}
        unit = ' HU' if self.is_ct() else ''
        clear_layout(self.stats_box); clear_layout(self.compare_box)
        self.stats_rows, self.compare_rows = [], []
        for st in result['stats']:
            if not st['voxels']:
                pairs = [('Thể tích', 'không có voxel')]
            else:
                pairs = [('Thể tích', fmt(st['ml'], '.2f') + ' mL' if st['ml'] is not None else 'n/a'),
                         ('Số khối rời nhau', fmt(st['components'], 'd')),
                         ('Đường kính lớn nhất', fmt(st['max_diameter_mm'], '.1f', '—') + ' mm'),
                         ('Cường độ TB ± SD', f"{st['mean']:.1f} ± {st['std']:.1f}{unit}")]
            self.stats_rows.append(dict(pairs))
            self.stats_box.addWidget(region_card(colors.get(st['label'], (255, 255, 255)), names.get(st['label'], str(st['label'])), pairs))
        for c in result['comparison'] or []:
            pairs = [('Dice', fmt(c['dice'], '.3f')), ('IoU', fmt(c['iou'], '.3f')), ('Chênh lệch thể tích', fmt(c['delta_ml'], '+.2f') + ' mL' if c['delta_ml'] is not None else 'n/a')]
            self.compare_rows.append(dict(pairs))
            self.compare_box.addWidget(region_card(colors.get(c['label'], (255, 255, 255)), names.get(c['label'], str(c['label'])), pairs))

    # ===== inference ===========================================================================
    def infer_clicked(self):
        if self.inference_worker is not None:
            self.inference_worker.cancel()
            self.infer.setEnabled(False); self.infer.setText(' Đang huỷ…')
            return
        self.run_inference()

    def run_inference(self):
        if self.volume is None or self.inference_worker is not None or self.infer_block_reason():
            return
        if self.mask_unsaved and self.mask is not None and not self.confirm_discard('Chạy AI sẽ thay thế mask hiện tại.'):
            return
        self.infer.setText(' Huỷ'); self.infer.setObjectName('danger'); self.infer.setIcon(QIcon())
        self.infer.style().unpolish(self.infer); self.infer.style().polish(self.infer)
        self.infer_progress.show()
        self.statusBar().showMessage('Đang chạy nnU-Net trên ảnh đang mở…')
        for view in self.views.values():
            view.canvas.spinner.start()
        worker = InferenceWorker(self.volume, self.current_task, self.want_uncertainty.isChecked())
        worker.done.connect(self.inference_done)
        worker.failed.connect(self.inference_failed)
        worker.notice.connect(lambda text: self.statusBar().showMessage(text))
        worker.cancelled.connect(lambda: self.statusBar().showMessage('Đã huỷ chạy phân đoạn.'))
        worker.finished.connect(self.inference_finished)
        self.inference_worker = worker
        self.update_infer_state()
        worker.start()

    def inference_done(self, mask_path, entropy):
        self._pending_heat = entropy
        self.load_mask(mask_path, is_ai_result=True)
        self.statusBar().showMessage('Đã chạy xong phân đoạn · kết quả AI, không phải nhãn tham chiếu')

    def inference_failed(self, message):
        self.statusBar().showMessage('Chạy phân đoạn thất bại.')
        QMessageBox.warning(self, 'Không chạy được phân đoạn', message)

    def inference_finished(self):
        self.inference_worker = None
        self.infer_progress.hide()
        self.infer.setObjectName('primary'); self.infer.setText(' Chạy AI'); self.infer.setIcon(QIcon(f'{ASSETS}/run.svg'))
        self.infer.style().unpolish(self.infer); self.infer.style().polish(self.infer)
        for view in self.views.values():
            view.canvas.spinner.stop()
        self.update_infer_state()

    # ===== export ==============================================================================
    def update_export_state(self):
        has_mask = self.volume is not None and self.mask is not None
        self.save_action.setEnabled(has_mask); self.mesh_action.setEnabled(has_mask and self.export_worker is None)
        self.report_action.setEnabled(has_mask)
        self.save_button.setEnabled(has_mask); self.clear_mask_button.setEnabled(has_mask)
        self.mask_button.setEnabled(self.volume is not None); self.ref_button.setEnabled(self.volume is not None)
        self.ref_clear.setEnabled(self.reference is not None)
        self.show_mask.setEnabled(has_mask)
        self.export_button.setEnabled(has_mask)

    def export_mesh(self):
        if self.volume is None or self.mask is None or self.export_worker is not None:
            return
        dialog = ExportMeshDialog(self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        options = dialog.options()
        folder = QFileDialog.getExistingDirectory(self, 'Chọn thư mục lưu mesh', str(self.volume.path.parent))
        if not folder:
            return
        base = strip_nifti(self.volume.path.name)
        names = {label_id: name for label_id, name, _ in self.entries}
        mask, affine, labels = self.mask.copy(), self.volume.image.affine.copy(), tuple(self.label_ids)

        def job():
            from .core.surfaces import build_surfaces
            meshes = build_surfaces(mask, affine, labels, reduction=0, smooth=options['smooth'])
            return export.export_meshes(meshes, names, folder, base, options['fmt'], options['per_label'], options['combined'])

        worker = FuncWorker(job)
        worker.result.connect(lambda files: self.statusBar().showMessage(f'Đã xuất {len(files)} file mesh vào {folder}'))
        worker.failed.connect(lambda message: QMessageBox.warning(self, 'Không xuất được mesh', message))
        worker.finished.connect(self.export_finished)
        self.export_worker = worker
        self.setCursor(Qt.CursorShape.BusyCursor)
        self.statusBar().showMessage('Đang dựng và ghi mesh…')
        self.update_export_state()
        worker.start()

    def export_finished(self):
        self.export_worker = None
        self.unsetCursor()
        self.update_export_state()

    def export_report(self):
        if self.volume is None or self.mask is None:
            return
        default = self.volume.path.parent / f'{strip_nifti(self.volume.path.name)}_report.json'
        path, selected = QFileDialog.getSaveFileName(self, 'Xuất báo cáo', str(default), 'JSON (*.json);;CSV (*.csv)')
        if not path:
            return
        if not path.lower().endswith(('.json', '.csv')):
            path += '.csv' if 'csv' in selected.lower() else '.json'
        self.write_report(path)

    def write_report(self, path):
        analysis = self.analysis or metrics.analyse(self.volume, max(0, self.channel.currentIndex()), self.mask, self.reference, tuple(self.label_ids))
        source = self.mask_source.text().splitlines()[0] if self.mask_source.text() else ''
        report = export.build_report(self.volume, self.current_task, self.mask, source, self.entries, analysis, self.edited)
        try:
            export.write_report(report, path)
        except Exception as error:
            QMessageBox.warning(self, 'Không xuất được báo cáo', str(error))
            return None
        self.statusBar().showMessage(f'Đã xuất báo cáo: {path}')
        return path

    # ===== lifecycle ===========================================================================
    def confirm_discard(self, what):
        answer = QMessageBox.question(self, 'Mask chưa được lưu', f'{what}\nMask hiện tại chưa được lưu. Tiếp tục?',
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel)
        return answer == QMessageBox.StandardButton.Yes

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.statusBar().showMessage('Đang đọc dữ liệu. Vui lòng đợi hoàn tất rồi đóng cửa sổ.')
            event.ignore(); return
        if self.surface_viewer and self.surface_viewer.is_busy():
            self.statusBar().showMessage('Đang dựng 3D, vui lòng đợi hoàn tất rồi đóng cửa sổ.')
            event.ignore(); return
        if self.export_worker is not None and self.export_worker.isRunning():
            self.statusBar().showMessage('Đang xuất mesh. Vui lòng đợi hoàn tất rồi đóng cửa sổ.')
            event.ignore(); return
        if self.mask is not None and self.mask_unsaved and not self.confirm_discard('Thoát OmniSeg.'):
            event.ignore(); return
        if self.inference_worker is not None:
            self.inference_worker.cancel(); self.inference_worker.wait(5000)
        if self.analysis_worker is not None:
            self.analysis_worker.wait(5000)
        if self.surface_viewer:
            self.surface_viewer.shutdown()
        event.accept()


def portal_available():
    try:
        from PySide6.QtDBus import QDBusConnection
        bus = QDBusConnection.sessionBus()
        return bus.isConnected() and bus.interface().isServiceRegistered('org.freedesktop.portal.Desktop').value()
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description='OmniSeg native desktop viewer')
    parser.add_argument('--file', type=Path, help='NIfTI file or DICOM folder to open')
    parser.add_argument('--mask', type=Path)
    parser.add_argument('--view', choices=['slices', '3d'], default='slices')
    args = parser.parse_args()
    if args.mask and not args.file:
        parser.error('--mask requires --file')
    # Never load the GTK theme plugin: some GNOME installs have incompatible GSettings
    # schemas. The xdg-desktop-portal theme gives native file dialogs over D-Bus instead;
    # fall back to Qt's own dialogs when no portal is running.
    if sys.platform.startswith('linux') and not os.environ.get('QT_QPA_PLATFORMTHEME'):
        os.environ['QT_QPA_PLATFORMTHEME'] = 'xdgdesktopportal' if portal_available() else 'generic'
    # VTK's native Qt interactor embeds an X11 OpenGL window on Linux.
    # Use XWayland on GNOME/Wayland; preserve explicit offscreen/test settings.
    if sys.platform.startswith('linux') and not os.environ.get('QT_QPA_PLATFORM'):
        os.environ['QT_QPA_PLATFORM'] = 'xcb'
    app = QApplication(sys.argv[:1]); app.setStyle('Fusion'); app.setStyleSheet(STYLE)
    app.setApplicationName('OmniSeg')
    window = Workspace(); window.show()
    if args.view == '3d':
        window.main = window.active = '3d'; window.set_layout('1x1')
    if args.file:
        def after_volume():
            if window.volume is not None and args.mask:
                window.load_mask(args.mask)
        window.load_volume(args.file)
        window.worker.finished.connect(lambda: QTimer.singleShot(0, after_volume))
    sys.exit(app.exec())
