"""Voxel uncertainty from nnU-Net's saved softmax probabilities."""
import numpy as np


def entropy_map(probabilities):
    """Normalised Shannon entropy (0 = certain, 1 = all classes equally likely), one value per voxel."""
    classes = probabilities.shape[0]
    result = np.zeros(probabilities.shape[1:], dtype=np.float32)
    # Slice by slice along the last axis: the full-resolution float array is large.
    for k in range(probabilities.shape[-1]):
        p = np.clip(probabilities[..., k].astype(np.float32), 1e-8, 1.0)
        result[..., k] = -(p * np.log(p)).sum(axis=0)
    return result / np.log(max(classes, 2))


def load_entropy(npz_path, volume):
    """Entropy map from nnU-Net's `<case>.npz`, laid out on the volume's canonical grid.

    nnU-Net stores arrays as (classes, z, y, x) for NIfTI input, i.e. reversed relative to
    the file's (x, y, z) axes; the shape decides which layout was used.
    """
    with np.load(npz_path) as archive:
        entropy = entropy_map(archive['probabilities'])
    file_shape = tuple(volume.orig_affine is not None and volume.to_original(volume.channels[0]).shape
                       or volume.channels[0].shape)
    if entropy.shape == file_shape[::-1] and entropy.shape != file_shape:
        entropy = entropy.transpose(2, 1, 0)
    if entropy.shape != file_shape:
        raise ValueError(f'Kích thước độ không chắc chắn {entropy.shape} không khớp ảnh {file_shape}.')
    return np.ascontiguousarray(volume.from_original(entropy).astype(np.float32))
