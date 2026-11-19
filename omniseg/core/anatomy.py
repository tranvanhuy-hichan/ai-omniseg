"""Volume rendering of the raw image in the same NIfTI world coordinates as mask surfaces."""
import numpy as np

# Fixed Hounsfield window for CT: soft tissue and air stay near the bottom of the
# 0..255 scale (and transparent); only bone rises into the opaque part of the ramp.
CT_WINDOW = (-200.0, 1000.0)
CT_AIR_HU = -500.0

_GRAY = [(0, (0, 0, 0)), (45, (.28,) * 3), (100, (.57,) * 3), (180, (.81,) * 3), (255, (.98,) * 3)]
PROFILES = {
    # Normalised-intensity MRI: opacity grows with intensity. Shading is off because
    # surface normals computed from MRI noise show up as fur on the translucent volume.
    'MRI': dict(
        colors=_GRAY,
        opacity=[(0, 0), (35, 0), (70, .008), (120, .035), (180, .10), (255, .24)],
        gain=1.0, gradient=None, shade=False),
    # CT: a faint skin shell for orientation plus bone. The gradient ramp hides the
    # homogeneous interior of soft tissue, which would otherwise build up into a fog
    # that covers the segmentation surfaces.
    'CT': dict(
        colors=[(0, (0, 0, 0)), (30, (.50, .50, .55)), (78, (.60, .58, .58)),
                (110, (.90, .87, .80)), (255, (1., .98, .92))],
        opacity=[(0, 0), (24, 0), (34, .30), (66, .30), (76, 0), (84, 0), (110, .5), (255, .9)],
        gain=3.0, gradient=[(0, 0), (8, 0), (30, 1), (255, 1)], shade=True),
}


def anatomy_kind(modality, raw):
    """'CT' or 'MRI'. The task modality wins; otherwise air at about -1000 HU means CT."""
    if str(modality or '').upper() == 'CT':
        return 'CT'
    return 'CT' if float(np.min(raw)) <= CT_AIR_HU else 'MRI'


def to_scalars(raw, kind, window=None):
    if kind == 'CT':
        low, high = CT_WINDOW
        return np.clip((raw - low) / (high - low) * 255, 0, 255).astype(np.uint8)
    low, high = window
    values = np.clip((raw - low) / max(high - low, 1e-6) * 255, 0, 255).astype(np.uint8)
    values[raw == 0] = 0
    return values


def smooth(values, passes=2):
    """Separable [1,2,1]/4 blur on a uint8 volume (numpy only, no extra dependency).

    MRI noise turns into streaks along the ray direction when volume-rendered; a light
    blur removes them. Used for the translucent context volume only, never for the mask."""
    out = values.astype(np.float32)
    for _ in range(passes):
        for axis in range(3):
            pad = [(1, 1) if a == axis else (0, 0) for a in range(3)]
            padded = np.pad(out, pad, mode='edge')
            def part(start, stop):
                index = [slice(None)] * 3; index[axis] = slice(start, stop); return padded[tuple(index)]
            out = (part(0, -2) + 2 * part(1, -1) + part(2, None)) / 4
    return np.clip(out + .5, 0, 255).astype(np.uint8)


def opacity_points(kind, strength):
    profile = PROFILES[kind]
    return [(value, min(1., amount * strength * profile['gain'])) for value, amount in profile['opacity']]


def create_anatomy(volume, channel, modality=None):
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPiecewiseFunction
    from vtkmodules.vtkCommonMath import vtkMatrix4x4
    from vtkmodules.vtkRenderingCore import vtkColorTransferFunction, vtkVolume, vtkVolumeProperty
    from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper

    raw = volume.channels[channel]
    kind = anatomy_kind(modality, raw)
    profile = PROFILES[kind]
    values = to_scalars(raw, kind, volume.window(channel) if kind == 'MRI' else None)
    # Remove empty background for a tighter camera fit. Keep the voxel offset
    # as image origin so anatomy and label surfaces remain in the same space.
    origin = [0,0,0]
    if np.any(values):
        occupied = values > 0
        limits = []
        for axis in range(3):
            indices = np.flatnonzero(np.any(occupied,axis=tuple(i for i in range(3) if i != axis)))
            origin[axis] = int(indices[0])
            limits.append(slice(int(indices[0]),int(indices[-1])+1))
        values = values[tuple(limits)]
    if kind == 'MRI': values = smooth(values)  # after the crop so world alignment stays exact
    image = vtkImageData(); image.SetDimensions(*values.shape)
    image.SetOrigin(*origin); image.SetSpacing(1,1,1)
    image.GetPointData().SetScalars(numpy_to_vtk(values.ravel(order='F'), deep=True))
    mapper = vtkSmartVolumeMapper(); mapper.SetInputData(image); mapper.SetBlendModeToComposite()
    # Ray-casting cost scales with sample count; auto-adjust so VTK coarsens
    # sampling while the camera is actively moving and sharpens once it settles,
    # instead of ray-casting at full quality on every frame of a drag.
    mapper.SetInteractiveUpdateRate(10)
    mapper.AutoAdjustSampleDistancesOn()
    colors = vtkColorTransferFunction()
    for intensity, shade in profile['colors']:
        colors.AddRGBPoint(intensity, *(shade if isinstance(shade, tuple) else (shade,) * 3))
    properties = vtkVolumeProperty(); properties.SetColor(colors); properties.SetInterpolationTypeToLinear()
    if profile['shade']:
        properties.ShadeOn(); properties.SetAmbient(.35); properties.SetDiffuse(.65); properties.SetSpecular(.1)
    else:
        properties.ShadeOff()
    if profile['gradient']:
        gradient = vtkPiecewiseFunction()
        for magnitude, amount in profile['gradient']: gradient.AddPoint(magnitude, amount)
        properties.SetGradientOpacity(gradient)
    matrix = vtkMatrix4x4()
    for i in range(4):
        for j in range(4): matrix.SetElement(i, j, float(volume.image.affine[i, j]))
    actor = vtkVolume(); actor.SetMapper(mapper); actor.SetProperty(properties); actor.SetUserMatrix(matrix)
    actor.omniseg_kind = kind
    set_anatomy_opacity(actor, .2)
    return actor


def set_anatomy_opacity(actor, strength):
    from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
    opacity = vtkPiecewiseFunction()
    for value, amount in opacity_points(getattr(actor, 'omniseg_kind', 'MRI'), strength):
        opacity.AddPoint(value, amount)
    actor.GetProperty().SetScalarOpacity(opacity)
