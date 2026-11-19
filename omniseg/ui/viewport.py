"""A PACS-style image viewport: black canvas, corner annotations, crosshair and mouse tools."""
import math

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QSlider, QWidget

from .theme import PLANE_COLORS

ORIENTATION = {2: ('L', 'R', 'A', 'P'), 1: ('L', 'R', 'S', 'I'), 0: ('P', 'A', 'S', 'I')}
PLANE_NAMES = {2: 'AXIAL', 1: 'CORONAL', 0: 'SAGITTAL'}
TOOLS = ('locate', 'window', 'pan', 'zoom', 'ruler', 'brush', 'eraser')


class Spinner(QWidget):
    """Indeterminate spinner overlaid on a viewport while something loads."""

    def __init__(self, parent=None, diameter=32):
        super().__init__(parent)
        self.diameter, self.angle = diameter, 0
        self.setFixedSize(diameter, diameter)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance)
        self.hide()

    def _advance(self):
        self.angle = (self.angle + 30) % 360
        self.update()

    def start(self):
        if self.parent() is not None:
            rect = self.parent().rect()
            self.move((rect.width() - self.width()) // 2, (rect.height() - self.height()) // 2)
        self.show(); self.raise_(); self.timer.start(80)

    def stop(self):
        self.timer.stop(); self.hide()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(4, 4, -4, -4)
        for i in range(12):
            color = QColor('#35bfcc')
            color.setAlphaF(max(0.08, 1 - i / 12))
            pen = QPen(color, max(3, self.diameter // 12))
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawArc(rect, (self.angle - i * 30) * 16, 16 * 18)


class SliceCanvas(QWidget):
    step = Signal(int)                     # wheel: +1 / -1 slices
    expand = Signal()                      # double click
    activated = Signal()                   # any press: this viewport becomes the active one
    locate = Signal(int, int)              # in-plane voxel under a left click / drag (locate tool)
    window_drag = Signal(float, float)     # dx, dy in pixels (W/L tool, or right button)
    paint = Signal(int, int, str)          # in-plane voxel, 'start' | 'move' | 'end'
    ruler_done = Signal(object, object)    # two (u, v) float voxel points
    hovered = Signal(object)               # (u, v) voxel under the cursor, or None

    def __init__(self, axis):
        super().__init__()
        self.axis, self.rgb, self.spacing = axis, None, (1., 1.)
        self.plane_shape = (1, 1)          # (n_u, n_v) voxels in this plane
        self.cursor_voxel = None           # (u, v) of the crosshair
        self.tool, self.show_cross, self.active = 'locate', True, False
        self.zoom, self.pan = 1.0, QPointF(0, 0)
        self.brush_mm, self.brush_color = 5.0, QColor('#ffffff')
        self.rulers, self._ruler_start, self._ruler_end = [], None, None
        self.corners = {'tl': [], 'tr': [], 'bl': [], 'br': []}
        self.banner, self.probe = '', ''
        self._drag, self._last, self._hover = None, None, None
        self.empty_text = 'Mở ảnh (Ctrl+O) để bắt đầu'
        self.setMouseTracking(True)
        self.setMinimumSize(160, 160)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.spinner = Spinner(self)

    # ----- geometry --------------------------------------------------------------------------
    def image_rect(self):
        n_u, n_v = self.plane_shape
        physical_w, physical_h = max(n_u * self.spacing[0], 1e-6), max(n_v * self.spacing[1], 1e-6)
        scale = min(max(1, self.width() - 12) / physical_w, max(1, self.height() - 12) / physical_h) * self.zoom
        w, h = physical_w * scale, physical_h * scale
        return QRectF((self.width() - w) / 2 + self.pan.x(), (self.height() - h) / 2 + self.pan.y(), w, h)

    def to_plane(self, pos):
        """Widget position -> float voxel coordinates (u, v); voxel centres are at k + 0.5."""
        rect, (n_u, n_v) = self.image_rect(), self.plane_shape
        return ((pos.x() - rect.x()) / rect.width() * n_u, n_v - (pos.y() - rect.y()) / rect.height() * n_v)

    def from_plane(self, u, v):
        rect, (n_u, n_v) = self.image_rect(), self.plane_shape
        return QPointF(rect.x() + u / n_u * rect.width(), rect.y() + (n_v - v) / n_v * rect.height())

    def voxel_at(self, pos):
        u, v = self.to_plane(pos)
        n_u, n_v = self.plane_shape
        if 0 <= u < n_u and 0 <= v < n_v:
            return int(u), int(v)
        return None

    def clamped_voxel(self, pos):
        u, v = self.to_plane(pos)
        n_u, n_v = self.plane_shape
        return min(max(int(math.floor(u)), 0), n_u - 1), min(max(int(math.floor(v)), 0), n_v - 1)

    def px_per_mm(self):
        rect = self.image_rect()
        return rect.width() / max(self.plane_shape[0] * self.spacing[0], 1e-6)

    def reset_view(self):
        self.zoom, self.pan = 1.0, QPointF(0, 0)
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.spinner.move((self.width() - self.spinner.width()) // 2, (self.height() - self.spinner.height()) // 2)

    # ----- painting --------------------------------------------------------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor('#000000'))
        if self.rgb is None:
            painter.setPen(QColor('#5d6b7b'))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.empty_text)
            self._frame(painter)
            return
        data = self.rgb
        image = QImage(data.data, data.shape[1], data.shape[0], data.strides[0], QImage.Format.Format_RGB888)
        rect = self.image_rect()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.zoom < 3)
        painter.drawImage(rect, image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setClipRect(self.rect())
        if self.show_cross and self.cursor_voxel is not None:
            color = QColor(PLANE_COLORS[self.axis]); color.setAlpha(150)
            painter.setPen(QPen(color, 1))
            point = self.from_plane(self.cursor_voxel[0] + .5, self.cursor_voxel[1] + .5)
            painter.drawLine(QPointF(rect.left(), point.y()), QPointF(rect.right(), point.y()))
            painter.drawLine(QPointF(point.x(), rect.top()), QPointF(point.x(), rect.bottom()))
        self._paint_rulers(painter)
        if self.tool in ('brush', 'eraser') and self._hover is not None:
            radius = self.brush_mm * self.px_per_mm()
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor('#ffffff') if self.tool == 'eraser' else self.brush_color, 1.4))
            painter.drawEllipse(self._hover, radius, radius)
        self._paint_text(painter)
        self._frame(painter)

    def _frame(self, painter):
        painter.setBrush(Qt.BrushStyle.NoBrush)
        color = QColor(PLANE_COLORS[self.axis]) if self.active else QColor('#2a3441')
        painter.setPen(QPen(color, 2 if self.active else 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

    def _paint_rulers(self, painter):
        lines = list(self.rulers)
        if self._ruler_start is not None and self._ruler_end is not None:
            lines.append((self._ruler_start, self._ruler_end))
        painter.setFont(self._font(12, bold=True))
        for start, end in lines:
            a, b = self.from_plane(*start), self.from_plane(*end)
            painter.setPen(QPen(QColor('#ffd24a'), 1.6))
            painter.drawLine(a, b)
            for point in (a, b):
                painter.drawEllipse(point, 3.5, 3.5)
            mm = math.hypot((end[0] - start[0]) * self.spacing[0], (end[1] - start[1]) * self.spacing[1])
            self._shadowed(painter, QPointF((a.x() + b.x()) / 2 + 8, (a.y() + b.y()) / 2 - 8), f'{mm:.1f} mm', '#ffd24a')

    @staticmethod
    def _font(size, bold=False):
        font = QFont(); font.setPixelSize(size); font.setBold(bold)
        return font

    @staticmethod
    def _shadowed(painter, point, text, color):
        painter.setPen(QColor(0, 0, 0, 220)); painter.drawText(point + QPointF(1, 1), text)
        painter.setPen(QColor(color)); painter.drawText(point, text)

    def _paint_text(self, painter):
        painter.setFont(self._font(12))
        pad, line = 9, 16
        color = '#e3ebf3'
        painter.setFont(self._font(12, bold=True))
        self._shadowed(painter, QPointF(pad, pad + 11), PLANE_NAMES[self.axis], PLANE_COLORS[self.axis])
        painter.setFont(self._font(12))
        for i, text in enumerate(self.corners['tl']):
            self._shadowed(painter, QPointF(pad, pad + 11 + line * (i + 1)), text, color)
        for i, text in enumerate(self.corners['tr']):
            self._shadowed(painter, QPointF(self.width() - pad - painter.fontMetrics().horizontalAdvance(text), pad + 11 + line * i), text, color)
        for i, text in enumerate(reversed(self.corners['bl'])):
            self._shadowed(painter, QPointF(pad, self.height() - pad - 3 - line * i), text, color)
        for i, text in enumerate(reversed(self.corners['br'] + ([self.probe] if self.probe else []))):
            self._shadowed(painter, QPointF(self.width() - pad - painter.fontMetrics().horizontalAdvance(text), self.height() - pad - 3 - line * i), text, color)
        left, right, top, bottom = ORIENTATION[self.axis]
        painter.setFont(self._font(14, bold=True))
        mid_x, mid_y = self.width() / 2, self.height() / 2
        self._shadowed(painter, QPointF(8, mid_y + 5), left, '#f2b84b')
        self._shadowed(painter, QPointF(self.width() - 20, mid_y + 5), right, '#f2b84b')
        self._shadowed(painter, QPointF(mid_x - 5, 26), top, '#f2b84b')
        self._shadowed(painter, QPointF(mid_x - 5, self.height() - 10), bottom, '#f2b84b')
        if self.banner:
            painter.setFont(self._font(12, bold=True))
            width = painter.fontMetrics().horizontalAdvance(self.banner) + 18
            box = QRectF((self.width() - width) / 2, 36, width, 22)
            painter.setPen(Qt.PenStyle.NoPen); painter.setBrush(QColor(122, 70, 12, 215))
            painter.drawRoundedRect(box, 4, 4)
            painter.setPen(QColor('#ffd9a0'))
            painter.drawText(box, Qt.AlignmentFlag.AlignCenter, self.banner)

    # ----- mouse -----------------------------------------------------------------------------
    def set_tool(self, tool):
        self.tool = tool
        self.setCursor({'pan': Qt.CursorShape.OpenHandCursor, 'zoom': Qt.CursorShape.SizeVerCursor,
                        'window': Qt.CursorShape.SizeAllCursor}.get(tool, Qt.CursorShape.CrossCursor))
        self.update()

    def mousePressEvent(self, event):
        self.activated.emit()
        if self.rgb is None:
            return
        self._last = event.position()
        button = event.button()
        if button == Qt.MouseButton.RightButton:
            self._drag = 'window'
        elif button == Qt.MouseButton.MiddleButton:
            self._drag = 'pan'
        elif button == Qt.MouseButton.LeftButton:
            self._drag = self.tool
        else:
            return
        if self._drag == 'locate':
            self.locate.emit(*self.clamped_voxel(event.position()))
        elif self._drag == 'ruler':
            self._ruler_start = self._ruler_end = self.to_plane(event.position())
        elif self._drag in ('brush', 'eraser'):
            voxel = self.voxel_at(event.position())
            if voxel is None:
                self._drag = None
            else:
                self.paint.emit(*voxel, 'start')
        if self._drag == 'pan':
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        self.update()

    def mouseMoveEvent(self, event):
        pos = event.position()
        self._hover = pos
        if self._drag is None or self.rgb is None:
            self.hovered.emit(self.voxel_at(pos))
            if self.tool in ('brush', 'eraser'):
                self.update()
            return
        delta = pos - self._last
        self._last = pos
        if self._drag == 'locate':
            self.locate.emit(*self.clamped_voxel(pos))
        elif self._drag == 'window':
            self.window_drag.emit(delta.x(), delta.y())
        elif self._drag == 'pan':
            self.pan += delta
        elif self._drag == 'zoom':
            self.zoom = min(max(self.zoom * 1.01 ** (-delta.y()), 0.25), 40.0)
        elif self._drag == 'ruler':
            self._ruler_end = self.to_plane(pos)
        elif self._drag in ('brush', 'eraser'):
            self.paint.emit(*self.clamped_voxel(pos), 'move')
        self.hovered.emit(self.voxel_at(pos))
        self.update()

    def mouseReleaseEvent(self, event):
        drag, self._drag = self._drag, None
        if drag == 'ruler' and self._ruler_start is not None:
            start, end = self._ruler_start, self._ruler_end
            self._ruler_start = self._ruler_end = None
            if start is not None and end is not None and math.hypot((end[0] - start[0]) * self.spacing[0], (end[1] - start[1]) * self.spacing[1]) > 0.5:
                self.ruler_done.emit(start, end)
        elif drag in ('brush', 'eraser'):
            self.paint.emit(0, 0, 'end')
        if drag == 'pan':
            self.set_tool(self.tool)
        self.update()

    def leaveEvent(self, event):
        self._hover = None
        self.hovered.emit(None)
        self.update()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.tool in ('locate', 'window', 'pan', 'zoom'):
            self.expand.emit()
            event.accept()

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom = min(max(self.zoom * (1.15 if event.angleDelta().y() > 0 else 1 / 1.15), 0.25), 40.0)
            self.update()
        else:
            self.step.emit(1 if event.angleDelta().y() > 0 else -1)
        event.accept()


class SliceView(QFrame):
    """Canvas plus a thin slice scrollbar on its right edge."""
    changed = Signal(int, int)

    def __init__(self, axis):
        super().__init__()
        self.axis = axis
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.canvas = SliceCanvas(axis)
        layout.addWidget(self.canvas, 1)
        self.slider = QSlider(Qt.Orientation.Vertical)
        self.slider.setFixedWidth(14)
        self.slider.setEnabled(False)
        self.slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.slider.valueChanged.connect(lambda value: self.changed.emit(axis, value))
        self.canvas.step.connect(lambda step: self.slider.setValue(self.slider.value() + step))
        layout.addWidget(self.slider)
        self.setStyleSheet('SliceView { background: #000; }')
