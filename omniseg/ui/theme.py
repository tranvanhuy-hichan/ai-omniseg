"""Dark, low-glare theme in the style of radiology workstations (PACS viewers)."""
from pathlib import Path

ASSETS = (Path(__file__).resolve().parents[1] / 'assets').as_posix()

BG, PANEL, TOOLBAR, BORDER = '#0e1318', '#151b22', '#1a212a', '#2a3441'
TEXT, MUTED, ACCENT, ACCENT_HOVER = '#d6dee7', '#8a9aac', '#1aa7b5', '#35bfcc'
PLANE_COLORS = {2: '#3d9bff', 1: '#41c27a', 0: '#e5a93c', '3d': '#b48cf2'}

STYLE = f'''
QMainWindow, QWidget {{ background: {BG}; color: {TEXT}; font-family: "Noto Sans", "Segoe UI", "Inter", sans-serif; font-size: 13px; }}
QLabel {{ background: transparent; }}
QToolTip {{ background: #232d3a; color: {TEXT}; border: 1px solid #3a4757; padding: 4px 6px; }}
QFrame#toolbar {{ background: {TOOLBAR}; border-bottom: 1px solid {BORDER}; }}
QFrame#toolbar QWidget {{ background: transparent; }}
QLabel#brand {{ color: white; font-size: 15px; font-weight: 700; letter-spacing: 0.5px; }}
QFrame#optionsbar {{ background: #131a21; border-bottom: 1px solid {BORDER}; }}
QFrame#optionsbar QWidget {{ background: transparent; }}
QFrame#optionsbar QComboBox, QFrame#optionsbar QDoubleSpinBox {{ background: #1b232d; }}
QFrame#separator {{ background: {BORDER}; max-width: 1px; min-width: 1px; }}
QToolButton {{ background: transparent; color: #c3cfdb; border: 1px solid transparent; border-radius: 5px; padding: 3px 6px; font-size: 11px; }}
QToolButton:hover {{ background: #263140; color: white; }}
QToolButton:checked {{ background: #17424d; border-color: {ACCENT}; color: white; }}
QToolButton:disabled {{ color: #4d5a69; }}
QToolButton::menu-indicator {{ image: none; width: 0; }}
QPushButton {{ background: #212a36; color: {TEXT}; border: 1px solid #344152; border-radius: 5px; padding: 6px 12px; }}
QPushButton:hover {{ background: #2a3544; border-color: #4a5a6e; }}
QPushButton:pressed {{ background: #1a222c; }}
QPushButton:disabled {{ background: #171d25; color: #55626f; border-color: #232c37; }}
QPushButton#primary {{ background: #0f8e9b; border: 1px solid #12a3b2; color: white; font-weight: 700; padding: 7px 12px; }}
QPushButton#primary:hover {{ background: #12a3b2; }}
QPushButton#primary:disabled {{ background: #17333a; border-color: #1d3f47; color: #5d8890; }}
QPushButton#danger {{ background: #7a2630; border-color: #9c3340; color: white; font-weight: 700; padding: 7px 12px; }}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{ background: #1b232d; border: 1px solid #344152; border-radius: 5px; padding: 4px 8px; selection-background-color: #17606b; min-height: 20px; }}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: #4a5a6e; }}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {ACCENT}; }}
QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{ color: #5a6775; background: #151b22; }}
QComboBox::drop-down {{ width: 24px; border: none; }}
QComboBox::down-arrow {{ image: url({ASSETS}/chevron_down.svg); width: 11px; height: 11px; }}
QComboBox QAbstractItemView {{ background: #1b232d; border: 1px solid #3a4757; outline: none; padding: 3px; selection-background-color: #17424d; selection-color: white; }}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 16px; border: none; background: transparent; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url({ASSETS}/chevron_up.svg); width: 9px; height: 9px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url({ASSETS}/chevron_down.svg); width: 9px; height: 9px; }}
QCheckBox {{ spacing: 10px; }}
QCheckBox::indicator {{ width: 36px; height: 20px; border: none; }}
QCheckBox::indicator:unchecked {{ image: url({ASSETS}/switch_off.svg); }}
QCheckBox::indicator:checked {{ image: url({ASSETS}/switch_on.svg); }}
QCheckBox::indicator:disabled {{ image: url({ASSETS}/switch_disabled.svg); }}
QSlider::groove:horizontal {{ height: 4px; background: #2c3745; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ height: 4px; background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; background: #e6edf4; border: none; }}
QSlider::handle:horizontal:hover {{ background: white; }}
QSlider::groove:vertical {{ width: 4px; background: #2c3745; border-radius: 2px; }}
QSlider::add-page:vertical {{ width: 4px; background: #2c3745; border-radius: 2px; }}
QSlider::sub-page:vertical {{ width: 4px; background: #2c3745; border-radius: 2px; }}
QSlider::handle:vertical {{ height: 12px; width: 12px; margin: 0 -4px; border-radius: 6px; background: #c9d4df; border: none; }}
QSlider::handle:vertical:hover {{ background: white; }}
QSlider:disabled::handle:vertical {{ background: #3a4655; }}
QTabWidget::pane {{ border: none; border-top: 1px solid {BORDER}; background: {PANEL}; }}
QTabBar {{ background: {PANEL}; }}
QTabBar::tab {{ background: {PANEL}; color: {MUTED}; padding: 9px 10px; border: none; border-bottom: 2px solid transparent; font-weight: 600; font-size: 12px; }}
QTabBar::tab:hover {{ color: white; }}
QTabBar::tab:selected {{ color: white; border-bottom: 2px solid {ACCENT}; }}
QScrollArea {{ border: none; background: {PANEL}; }}
QScrollArea > QWidget > QWidget {{ background: {PANEL}; }}
QScrollBar:vertical {{ width: 9px; background: transparent; }}
QScrollBar::handle:vertical {{ background: #38465a; min-height: 30px; border-radius: 4px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QLabel#section {{ color: {MUTED}; font-size: 11px; font-weight: 700; letter-spacing: 1px; padding-top: 6px; }}
QLabel#muted {{ color: {MUTED}; font-size: 12px; }}
QLabel#note {{ color: #b9a36a; font-size: 11.5px; }}
QGroupBox {{ background: #1a222c; border: 1px solid {BORDER}; border-radius: 7px; margin-top: 14px; padding: 16px 12px 12px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 5px; color: {MUTED}; font-size: 11px; letter-spacing: 1px; }}
QGroupBox QLabel, QGroupBox QCheckBox {{ background: transparent; }}
QFrame#card {{ background: #1a222c; border: 1px solid {BORDER}; border-radius: 7px; }}
QFrame#card QLabel, QFrame#card QCheckBox {{ background: transparent; }}
QTableWidget {{ background: #171e27; border: 1px solid {BORDER}; border-radius: 6px; gridline-color: #232d39; outline: none; }}
QTableWidget::item {{ padding: 4px 6px; }}
QHeaderView::section {{ background: #1d2631; color: {MUTED}; border: none; border-bottom: 1px solid {BORDER}; padding: 5px 6px; font-size: 11px; font-weight: 700; }}
QListWidget {{ background: #171e27; border: 1px solid {BORDER}; border-radius: 6px; outline: none; }}
QListWidget::item {{ padding: 5px 8px; }}
QListWidget::item:selected {{ background: #17424d; color: white; }}
QProgressBar {{ background: #243040; border: none; border-radius: 2px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 2px; }}
QStatusBar {{ background: {TOOLBAR}; border-top: 1px solid {BORDER}; color: {MUTED}; font-size: 12px; }}
QStatusBar::item {{ border: none; }}
QSplitter::handle {{ background: {BORDER}; }}
QMenu {{ background: #1b232d; border: 1px solid #3a4757; padding: 4px; }}
QMenu::item {{ padding: 6px 22px 6px 12px; border-radius: 4px; }}
QMenu::item:selected {{ background: #17424d; color: white; }}
QMenu::item:disabled {{ color: #55626f; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 6px; }}
QMessageBox, QDialog {{ background: {PANEL}; }}
'''
