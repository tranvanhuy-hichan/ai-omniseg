"""Small modal dialogs."""
from PySide6.QtWidgets import QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QVBoxLayout


class ExportMeshDialog(QDialog):
    """Choose mesh format and layout for 3D export."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Xuất mesh 3D')
        layout = QVBoxLayout(self)
        note = QLabel('Bề mặt của từng vùng phân đoạn, tính theo toạ độ thế giới (mm, hệ RAS) của ảnh.')
        note.setWordWrap(True); note.setObjectName('muted')
        layout.addWidget(note)
        form = QFormLayout()
        self.format = QComboBox()
        for key, text in (('stl', 'STL (in 3D, nhị phân)'), ('ply', 'PLY'), ('obj', 'OBJ')):
            self.format.addItem(text, key)
        self.per_label = QCheckBox('Mỗi vùng một file'); self.per_label.setChecked(True)
        self.combined = QCheckBox('Thêm một file gộp tất cả vùng'); self.combined.setChecked(True)
        self.smooth = QCheckBox('Làm mịn bề mặt (loại bậc thang voxel)'); self.smooth.setChecked(True)
        form.addRow('Định dạng', self.format)
        layout.addLayout(form)
        for box in (self.per_label, self.combined, self.smooth):
            layout.addWidget(box)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def options(self):
        return {'fmt': self.format.currentData(), 'per_label': self.per_label.isChecked(),
                'combined': self.combined.isChecked(), 'smooth': 15 if self.smooth.isChecked() else 0}
