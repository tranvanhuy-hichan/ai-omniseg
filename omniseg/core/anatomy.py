"""MRI volume rendering in the same NIfTI world coordinates as mask surfaces."""
import numpy as np


def create_anatomy(volume, channel):
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonDataModel import vtkImageData
    from vtkmodules.vtkCommonMath import vtkMatrix4x4
    from vtkmodules.vtkRenderingCore import vtkColorTransferFunction, vtkVolume, vtkVolumeProperty
    from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper

    low, high = volume.window(channel)
    raw = volume.channels[channel]
    values = np.clip((raw-low)/max(high-low,1e-6)*255,0,255).astype(np.uint8)
    values[raw == 0] = 0
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
    image = vtkImageData(); image.SetDimensions(*values.shape)
    image.SetOrigin(*origin); image.SetSpacing(1,1,1)
    image.GetPointData().SetScalars(numpy_to_vtk(values.ravel(order='F'), deep=True))
    mapper = vtkSmartVolumeMapper(); mapper.SetInputData(image); mapper.SetBlendModeToComposite()
    # Ray-casting cost scales with sample count; auto-adjust so VTK coarsens
    # sampling while the camera is actively moving and sharpens once it settles,
    # instead of ray-casting at full quality on every frame of a drag.
    mapper.SetInteractiveUpdateRate(10)
    mapper.AutoAdjustSampleDistancesOn()
    colors=vtkColorTransferFunction()
    for intensity,shade in [(0,0),(45,.28),(100,.57),(180,.81),(255,.98)]: colors.AddRGBPoint(intensity,shade,shade,shade)
    properties=vtkVolumeProperty(); properties.SetColor(colors);properties.SetInterpolationTypeToLinear()
    properties.ShadeOn();properties.SetAmbient(.35);properties.SetDiffuse(.65);properties.SetSpecular(.1)
    matrix=vtkMatrix4x4()
    for i in range(4):
        for j in range(4):matrix.SetElement(i,j,float(volume.image.affine[i,j]))
    actor=vtkVolume();actor.SetMapper(mapper);actor.SetProperty(properties);actor.SetUserMatrix(matrix)
    set_anatomy_opacity(actor, .2)
    return actor


def set_anatomy_opacity(actor, strength):
    from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
    opacity=vtkPiecewiseFunction()
    for value,amount in [(0,0),(35,0),(70,.008),(120,.035),(180,.10),(255,.24)]:
        opacity.AddPoint(value,amount*strength)
    actor.GetProperty().SetScalarOpacity(opacity)
