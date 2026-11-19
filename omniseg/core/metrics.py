"""Mask comparison and per-region measurements. NumPy only; scipy is used when installed."""
import numpy as np

CT_PRESETS = [('Mô mềm', 40, 400), ('Gan', 60, 160), ('Phổi', -600, 1500), ('Xương', 300, 1500), ('Não', 40, 80)]


def compare_masks(prediction, reference, labels, voxel_ml=None):
    """Per-label Dice / IoU between two label masks of the same shape.

    A label absent from both masks has no defined overlap, so Dice and IoU are None (shown
    as "n/a"); a label present in only one of them scores 0.0.
    """
    if prediction.shape != reference.shape:
        raise ValueError('Hai mask phải cùng kích thước.')
    rows = []
    for label in labels:
        pred, ref = prediction == label, reference == label
        pred_count, ref_count = int(np.count_nonzero(pred)), int(np.count_nonzero(ref))
        inter = int(np.count_nonzero(pred & ref))
        union = pred_count + ref_count - inter
        row = {
            'label': int(label),
            'dice': None if pred_count + ref_count == 0 else 2 * inter / (pred_count + ref_count),
            'iou': None if union == 0 else inter / union,
            'prediction_voxels': pred_count, 'reference_voxels': ref_count,
            'prediction_ml': None, 'reference_ml': None, 'delta_ml': None,
        }
        if voxel_ml is not None:
            row['prediction_ml'], row['reference_ml'] = pred_count * voxel_ml, ref_count * voxel_ml
            row['delta_ml'] = row['prediction_ml'] - row['reference_ml']
        rows.append(row)
    return rows


def max_diameter_mm(binary, spacing, max_points=4000):
    """Largest distance between two voxel centres of `binary`, in mm (exact via the convex hull)."""
    points = np.argwhere(binary)
    if len(points) < 2:
        return 0.0
    points = points.astype(float) * np.asarray(spacing, dtype=float)
    if len(points) > max_points:
        try:
            from scipy.spatial import ConvexHull
            points = points[ConvexHull(points).vertices]
        except Exception:
            points = points[np.random.default_rng(0).choice(len(points), max_points, replace=False)]
    if len(points) > max_points:
        points = points[np.random.default_rng(0).choice(len(points), max_points, replace=False)]
    best = 0.0
    for start in range(0, len(points), 512):
        chunk = points[start:start + 512]
        best = max(best, float(np.sqrt(((chunk[:, None, :] - points[None, :, :]) ** 2).sum(-1).max())))
    return best


def region_stats(mask, label, image, spacing, voxel_ml=None):
    """Volume, connected components, largest-component diameter and intensity of one label."""
    binary = mask == label
    voxels = int(np.count_nonzero(binary))
    result = {'label': int(label), 'voxels': voxels, 'ml': None if voxel_ml is None else voxels * voxel_ml,
              'components': 0, 'max_diameter_mm': None, 'mean': None, 'std': None}
    if not voxels:
        return result
    values = image[binary]
    result['mean'], result['std'] = float(values.mean()), float(values.std())
    # Work inside the bounding box: lesions are tiny compared with the volume.
    coords = np.argwhere(binary)
    low, high = coords.min(0), coords.max(0) + 1
    crop = binary[tuple(slice(a, b) for a, b in zip(low, high))]
    try:
        from scipy import ndimage
        components, count = ndimage.label(crop, structure=np.ones((3, 3, 3)))
        sizes = np.bincount(components.ravel())[1:]
        result['components'] = int(count)
        crop = components == (int(np.argmax(sizes)) + 1)
        crop = crop & ~ndimage.binary_erosion(crop)
    except ImportError:
        result['components'] = None
    result['max_diameter_mm'] = max_diameter_mm(crop, spacing)
    return result


def analyse(volume, channel, mask, reference, labels):
    """Everything the analysis panel shows for the current mask (and optional reference)."""
    image = volume.channels[channel]
    ml = volume.voxel_ml
    stats = [region_stats(mask, label, image, volume.spacing, ml) for label in labels]
    comparison = compare_masks(mask, reference, labels, ml) if reference is not None else None
    return {'stats': stats, 'comparison': comparison}
