"""Label surfaces in NIfTI world coordinates; no rendering/Qt dependency here."""
import numpy as np


def build_surfaces(mask, affine):
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonMath import vtkMatrix4x4
    from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPolyData
    from vtkmodules.vtkCommonTransforms import vtkTransform
    from vtkmodules.vtkFiltersCore import vtkPolyDataNormals, vtkDecimatePro
    from vtkmodules.vtkFiltersGeneral import vtkTransformFilter, vtkDiscreteFlyingEdges3D

    values = np.asarray(mask)
    affine = np.asarray(affine, dtype=float)
    if values.ndim != 3 or not np.isin(values, [0, 1, 2, 3]).all():
        raise ValueError('Mask 3D phải có nhãn 0–3.')
    if affine.shape != (4, 4) or not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3,:3])) < 1e-12:
        raise ValueError('Affine không hợp lệ.')
    # Pad background to close surfaces even when foreground touches an image edge.
    # VTK x varies fastest; the source NumPy array is indexed [x,y,z].
    padded = np.pad(values.astype(np.uint8), 1)
    image = vtkImageData()
    image.SetDimensions(*padded.shape)
    image.SetOrigin(-1, -1, -1)
    image.SetSpacing(1, 1, 1)
    image.GetPointData().SetScalars(numpy_to_vtk(padded.ravel(order='F'), deep=True))
    matrix = vtkMatrix4x4()
    for row in range(4):
        for column in range(4): matrix.SetElement(row,column,float(affine[row,column]))
    transform = vtkTransform(); transform.SetMatrix(matrix)
    result = {}
    for label in (1, 2, 3):
        if not np.any(values == label): continue
        contour = vtkDiscreteFlyingEdges3D()
        contour.SetInputData(image); contour.SetValue(0, label)
        contour.ComputeNormalsOff(); contour.ComputeGradientsOff()
        world = vtkTransformFilter(); world.SetTransform(transform); world.SetInputConnection(contour.GetOutputPort())
        # Flying edges on a ~240^3 volume produces far more triangles than the
        # screen can resolve; decimating keeps rotation/zoom smooth without a
        # visibly different silhouette.
        decimate = vtkDecimatePro(); decimate.SetInputConnection(world.GetOutputPort())
        decimate.SetTargetReduction(0.6); decimate.PreserveTopologyOn()
        normals = vtkPolyDataNormals(); normals.SetInputConnection(decimate.GetOutputPort())
        normals.ConsistencyOn(); normals.AutoOrientNormalsOn(); normals.SplittingOff(); normals.Update()
        surface = vtkPolyData(); surface.DeepCopy(normals.GetOutput())
        if surface.GetNumberOfPoints(): result[label] = surface
    return result
