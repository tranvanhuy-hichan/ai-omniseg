"""NIfTI loading, canonical orientation and slice rendering, independent of Qt."""
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

CHANNELS = ['FLAIR', 'T1w', 'T1 + Gd', 'T2w']
COLORS = np.array([[0, 0, 0], [245, 180, 69], [56, 189, 181], [231, 99, 138]], dtype=np.uint8)


@dataclass
class Volume:
    path: Path
    image: object
    data: np.ndarray
    names: list[str]
    oblique: bool
    channels: list = None

    def __post_init__(self):
        # nibabel returns Fortran-ordered arrays; slicing a channel's 3D volume out
        # of that (or out of a plain [..., channel] view) along any axis is very
        # slow (non-contiguous access). Cache each channel as its own C-contiguous
        # 3D array once, instead of paying that cost on every slider move.
        if self.channels is None:
            self.channels = [np.ascontiguousarray(self.data[..., c]) for c in range(self.data.shape[3])]

    @classmethod
    def load(cls, path):
        path = Path(path)
        image = nib.as_closest_canonical(nib.load(path))
        if len(image.shape) not in (3, 4) or any(s < 1 for s in image.shape):
            raise ValueError('Cần NIfTI 3D hoặc 4D có kích thước hợp lệ.')
        if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-12:
            raise ValueError('Affine không hợp lệ.')
        data = image.get_fdata(dtype=np.float32)
        if not np.isfinite(data).all():
            raise ValueError('MRI có giá trị NaN/Inf. Hãy kiểm tra dữ liệu đầu vào.')
        if data.ndim == 3:
            data = data[..., None]
        known_msd = any(part in ('Task01_BrainTumour', 'Dataset501_BrainTumour') for part in path.parts)
        names = CHANNELS.copy() if known_msd and data.shape[3] == 4 else [f'Kênh {i + 1}' for i in range(data.shape[3])]
        stem = path.name.removesuffix('.nii.gz').removesuffix('.nii')
        if known_msd and len(names) == 1 and stem[-5:] in [f'_{i:04d}' for i in range(4)]:
            names = [CHANNELS[int(stem[-4:])]]
        axes = image.affine[:3, :3]
        oblique = not np.allclose(axes, np.diag(np.diag(axes)), atol=1e-4)
        return cls(path, image, data, names, oblique)

    @property
    def spacing(self):
        return np.array(self.image.header.get_zooms()[:3])

    def window(self, channel):
        values = self.channels[channel]
        values = values[values != 0]
        if not values.size:
            return (0., 1.)
        low, high = np.percentile(values[::max(1, values.size // 200000)], [1, 99])
        return float(low), float(max(high, low + 1))

    def load_mask(self, path, labels=(1, 2, 3)):
        mask = nib.as_closest_canonical(nib.load(path))
        if mask.shape != self.data.shape[:3] or not np.allclose(mask.affine, self.image.affine, atol=1e-4, rtol=0):
            raise ValueError('Mask không cùng shape/affine với ảnh. Cần đăng ký ảnh trước khi chồng mask.')
        values = mask.get_fdata(dtype=np.float32)
        allowed = [0, *labels]
        if not np.isin(values, allowed).all():
            raise ValueError(f'Mask có nhãn ngoài tập của mô hình đang chọn (chỉ nhận nhãn {allowed}).')
        return np.ascontiguousarray(values.astype(np.uint8))

    def label_volumes(self, mask, labels=(1, 2, 3)):
        unit = self.image.header.get_xyzt_units()[0]
        factor = {'mm': 0.001, 'meter': 1e6, 'micron': 1e-12}.get(unit)
        if factor is None:
            return None
        voxel_ml = abs(np.linalg.det(self.image.affine[:3, :3])) * factor
        return [float(np.count_nonzero(mask == label) * voxel_ml) for label in labels]


def plane(array, axis, index):
    """RAS voxel planes, neurological display: L on the left of axial/coronal."""
    return np.ascontiguousarray(np.flipud(np.take(array, index, axis=axis).T))


def color_table(entries):
    """RGB lookup indexed by label id from registry.label_entries(); unlisted ids stay black."""
    table = np.zeros((max([label for label, _, _ in entries], default=0) + 1, 3), dtype=np.uint8)
    for label, _, color in entries:
        table[label] = color
    return table


def render_slice(volume, axis, position, channel, window, mask=None, opacity=0.35, visible_labels=None, colors=None):
    values = plane(volume.channels[channel], axis, position)
    low, high = window
    gray = np.clip((values - low) / max(high - low, 1e-6) * 255, 0, 255).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    if mask is not None:
        labels = plane(mask, axis, position)
        if visible_labels is not None:
            labels = np.where(np.isin(labels, visible_labels), labels, 0)
        table = COLORS if colors is None else colors
        foreground = (labels > 0) & (labels < len(table))
        rgb[foreground] = ((1 - opacity) * rgb[foreground] + opacity * table[labels[foreground]]).astype(np.uint8)
    return np.ascontiguousarray(rgb)
