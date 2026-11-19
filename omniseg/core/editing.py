"""Brush editing of a label mask with undo/redo. NumPy only."""
import numpy as np


def plane_view(mask, axis, index):
    """Writable 2D view of one slice; its axes are the two remaining mask axes in order."""
    selector = [slice(None)] * 3
    selector[axis] = index
    return mask[tuple(selector)]


def apply_brush(mask, axis, index, center, radius_mm, spacing, value, only=None):
    """Paint a disc of `radius_mm` at voxel `center` (the two in-plane indices) into one slice.

    value: label to write (0 erases). only: if given, only voxels that currently hold this
    label are changed (so the eraser removes one label without touching the others).
    Returns True when any voxel changed.
    """
    view = plane_view(mask, axis, index)
    remaining = [i for i in range(3) if i != axis]
    s0, s1 = (float(spacing[i]) for i in remaining)
    c0, c1 = center
    r0, r1 = int(np.ceil(radius_mm / s0)), int(np.ceil(radius_mm / s1))
    a0, b0 = max(0, c0 - r0), min(view.shape[0], c0 + r0 + 1)
    a1, b1 = max(0, c1 - r1), min(view.shape[1], c1 + r1 + 1)
    if a0 >= b0 or a1 >= b1:
        return False
    u = (np.arange(a0, b0) - c0)[:, None] * s0
    v = (np.arange(a1, b1) - c1)[None, :] * s1
    disc = u * u + v * v <= radius_mm * radius_mm
    region = view[a0:b0, a1:b1]
    if only is not None:
        disc = disc & (region == only)
    before = region.copy()
    region[disc] = value
    return bool(np.any(region != before))


class MaskHistory:
    """Stroke-level undo/redo storing the changed voxels (flat index, old value, new value)."""

    def __init__(self, limit=60):
        self.limit = limit
        self.undo_stack, self.redo_stack, self._stroke = [], [], None

    def begin(self):
        self._stroke = {}

    def edit(self, mask, axis, index, change):
        """Run `change()` (which modifies slice (axis, index) of `mask`) and record what it altered.

        The first old value seen for a voxel wins, so strokes crossing several planes undo exactly.
        """
        before = plane_view(mask, axis, index).copy()
        result = change()
        if self._stroke is not None:
            view = plane_view(mask, axis, index)
            changed = np.argwhere(view != before)
            for position in changed:
                full = list(map(int, position))
                full.insert(axis, int(index))
                flat = int(np.ravel_multi_index(full, mask.shape))
                self._stroke.setdefault(flat, int(before[tuple(position)]))
        return result

    def commit(self, mask):
        stroke, self._stroke = self._stroke, None
        if not stroke:
            return False
        flat = np.fromiter(stroke.keys(), dtype=np.int64, count=len(stroke))
        old = np.fromiter(stroke.values(), dtype=mask.dtype, count=len(stroke))
        self.undo_stack.append((flat, old, mask.reshape(-1)[flat].copy()))
        del self.undo_stack[:-self.limit]
        self.redo_stack.clear()
        return True

    def undo(self, mask):
        if not self.undo_stack:
            return False
        record = self.undo_stack.pop()
        mask.reshape(-1)[record[0]] = record[1]
        self.redo_stack.append(record)
        return True

    def redo(self, mask):
        if not self.redo_stack:
            return False
        record = self.redo_stack.pop()
        mask.reshape(-1)[record[0]] = record[2]
        self.undo_stack.append(record)
        return True

    def clear(self):
        self.undo_stack.clear(); self.redo_stack.clear(); self._stroke = None

    @property
    def can_undo(self):
        return bool(self.undo_stack)

    @property
    def can_redo(self):
        return bool(self.redo_stack)
