"""Label surfaces in NIfTI world coordinates; no rendering/Qt dependency here."""
import numpy as np


def build_surfaces(mask, affine, labels=(1, 2, 3), reduction=0.6, smooth=0):
    """Label surfaces in world mm. `reduction` is the decimation fraction (0 keeps every triangle);
    `smooth` is the number of windowed-sinc smoothing passes (0 = none)."""
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonMath import vtkMatrix4x4
    from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPolyData
    from vtkmodules.vtkCommonTransforms import vtkTransform
    from vtkmodules.vtkFiltersCore import vtkPolyDataNormals, vtkDecimatePro, vtkWindowedSincPolyDataFilter
    from vtkmodules.vtkFiltersGeneral import vtkTransformFilter, vtkDiscreteFlyingEdges3D

    values = np.asarray(mask)
    affine = np.asarray(affine, dtype=float)
    allowed = [0, *labels]
    if values.ndim != 3 or not np.isin(values, allowed).all():
        raise ValueError(f'Mask 3D chỉ được có nhãn {allowed}.')
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
    for label in labels:
        if not np.any(values == label): continue
        contour = vtkDiscreteFlyingEdges3D()
        contour.SetInputData(image); contour.SetValue(0, label)
        contour.ComputeNormalsOff(); contour.ComputeGradientsOff()
        world = vtkTransformFilter(); world.SetTransform(transform); world.SetInputConnection(contour.GetOutputPort())
        # Flying edges on a ~240^3 volume produces far more triangles than the
        # screen can resolve; decimating keeps rotation/zoom smooth without a
        # visibly different silhouette.
        source = world
        if reduction > 0:
            decimate = vtkDecimatePro(); decimate.SetInputConnection(world.GetOutputPort())
            decimate.SetTargetReduction(reduction); decimate.PreserveTopologyOn()
            source = decimate
        if smooth > 0:
            smoother = vtkWindowedSincPolyDataFilter(); smoother.SetInputConnection(source.GetOutputPort())
            smoother.SetNumberOfIterations(int(smooth)); smoother.BoundarySmoothingOff(); smoother.NonManifoldSmoothingOn()
            smoother.NormalizeCoordinatesOn()
            source = smoother
        normals = vtkPolyDataNormals(); normals.SetInputConnection(source.GetOutputPort())
        normals.ConsistencyOn(); normals.AutoOrientNormalsOn(); normals.SplittingOff(); normals.Update()
        surface = vtkPolyData(); surface.DeepCopy(normals.GetOutput())
        if surface.GetNumberOfPoints(): result[label] = surface
    return result
