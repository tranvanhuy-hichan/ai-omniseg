"""Optional VTK geometry tests; no display or OpenGL is required."""
import numpy as np
import pytest

pytest.importorskip('vtkmodules')
from desktop.surfaces import build_surfaces


def test_world_coordinates_include_rotation_spacing_and_origin():
    mask=np.zeros((3,4,5),dtype=np.uint8);mask[1,2,3]=2
    affine=np.array([[0,-3,0,10],[2,0,0,20],[0,0,4,30],[0,0,0,1]],dtype=float)
    surfaces=build_surfaces(mask,affine)
    assert set(surfaces)=={2}
    assert surfaces[2].GetNumberOfCells()>0
    assert surfaces[2].GetBounds()==pytest.approx((2.5,5.5,21,23,40,44))


def test_edge_touching_label_is_closed_and_empty_labels_are_absent():
    from vtkmodules.vtkFiltersCore import vtkFeatureEdges
    mask=np.zeros((3,3,3),dtype=np.uint8);mask[0,0,0]=1
    surface=build_surfaces(mask,np.eye(4))[1]
    assert surface.GetBounds()==pytest.approx((-.5,.5,-.5,.5,-.5,.5))
    edges=vtkFeatureEdges();edges.SetInputData(surface);edges.BoundaryEdgesOn();edges.FeatureEdgesOff()
    edges.NonManifoldEdgesOff();edges.ManifoldEdgesOff();edges.Update()
    assert edges.GetOutput().GetNumberOfCells()==0
    assert build_surfaces(np.zeros((3,3,3),dtype=np.uint8),np.eye(4))=={}


def test_mri_rendering_crop_preserves_world_alignment(tmp_path):
    import nibabel as nib
    from desktop.volume import Volume
    from desktop.anatomy import create_anatomy, set_anatomy_opacity
    data=np.zeros((10,11,12),dtype=np.float32)
    data[2:6,3:7,4:8]=np.arange(64,dtype=np.float32).reshape(4,4,4)+100
    affine=np.diag([2.,3.,4.,1.]);affine[:3,3]=[10,20,30]
    path=tmp_path/'mri.nii.gz';nib.save(nib.Nifti1Image(data,affine),path)
    actor=create_anatomy(Volume.load(path),0)
    assert actor.GetBounds()==pytest.approx((14,20,29,38,46,58))
    set_anatomy_opacity(actor,.5)
    assert actor.GetProperty().GetScalarOpacity().GetValue(0)==0
