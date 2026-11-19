"""Background threads: loading, nnU-Net inference and generic jobs."""
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
from PySide6.QtCore import QThread, Signal

from ..core import inference, uncertainty
from ..core.export import strip_nifti
from ..core.volume import Volume


class Loader(QThread):
    """Loads an image (NIfTI file or DICOM folder) or, given `volume`, a mask for it."""
    loaded = Signal(object)
    failed = Signal(str)

    def __init__(self, path, volume=None, labels=(1, 2, 3)):
        super().__init__()
        self.path, self.volume, self.labels = path, volume, labels

    def run(self):
        try:
            if self.volume is not None:
                self.loaded.emit(self.volume.load_mask(self.path, self.labels))
                return
            path = Path(self.path)
            if path.is_dir():
                from ..core import dicom
                path = dicom.convert_series(path)
            self.loaded.emit(Volume.load(path))
        except Exception as error:
            self.failed.emit(str(error))


class FuncWorker(QThread):
    """Runs fn(*args) off the UI thread; emits the result or the error text."""
    result = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args):
        super().__init__()
        self.fn, self.args = fn, args

    def run(self):
        try:
            self.result.emit(self.fn(*self.args))
        except Exception as error:
            self.failed.emit(str(error))


class InferenceWorker(QThread):
    """Runs nnUNetv2_predict for `task` on the open volume's channels in a temp folder."""
    done = Signal(str, object)     # mask path, entropy map (or None)
    failed = Signal(str)
    notice = Signal(str)
    cancelled = Signal()

    def __init__(self, volume, task, want_uncertainty=False):
        super().__init__()
        self.volume, self.task, self.want_uncertainty = volume, task, want_uncertainty
        self._process, self._cancel = None, False

    def cancel(self):
        self._cancel = True
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()

    def run(self):
        try:
            with tempfile.TemporaryDirectory(prefix='omniseg-infer-') as tmp:
                tmp = Path(tmp)
                input_dir, output_dir = tmp / 'input', tmp / 'output'
                input_dir.mkdir(); output_dir.mkdir()
                case = strip_nifti(self.volume.path.name)
                data, affine = self.volume.data, self.volume.image.affine
                for channel in range(data.shape[3]):
                    nib.save(nib.Nifti1Image(data[..., channel], affine), input_dir / f'{case}_{channel:04d}.nii.gz')
                command = inference.build_predict_command(self.task, input_dir, output_dir, self.want_uncertainty)
                self._process = subprocess.Popen(command, env=inference.build_predict_env(self.task),
                                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                _, stderr = self._process.communicate()
                if self._cancel:
                    self.cancelled.emit()
                    return
                if self._process.returncode != 0:
                    self.failed.emit((stderr or 'nnUNetv2_predict thất bại.')[-2000:])
                    return
                predicted = output_dir / f'{case}.nii.gz'
                if not predicted.is_file():
                    raise FileNotFoundError('Không tìm thấy mask dự đoán sau khi chạy xong.')
                destination = Path(tempfile.gettempdir()) / f'omniseg-prediction-{case}.nii.gz'
                destination.write_bytes(predicted.read_bytes())
                entropy = None
                if self.want_uncertainty:
                    try:
                        entropy = uncertainty.load_entropy(output_dir / f'{case}.npz', self.volume)
                    except Exception as error:
                        self.notice.emit(f'Không tính được độ không chắc chắn: {error}')
                self.done.emit(str(destination), entropy)
        except Exception as error:
            self.failed.emit(str(error))
