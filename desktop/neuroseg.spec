# PyInstaller spec for the NeuroSeg desktop viewer.
# Build with: pyinstaller desktop/neuroseg.spec --distpath dist --workpath build
from pathlib import Path

block_cipher = None
root = Path(SPECPATH).resolve().parent

a = Analysis(
    [str(root / 'desktop' / 'entry_point.py')],
    pathex=[str(root)],
    binaries=[],
    datas=[(str(root / 'desktop' / 'assets'), 'desktop/assets')],
    hiddenimports=[
        'vtkmodules.all',
        'vtkmodules.util.numpy_support',
        'vtkmodules.qt.QVTKRenderWindowInteractor',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='NeuroSeg',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='NeuroSeg',
)
