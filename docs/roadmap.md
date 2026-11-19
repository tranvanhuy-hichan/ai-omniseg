# Roadmap

## Available

- [x] Standalone repository with nnU-Net as a dependency
- [x] Configurable NIfTI importer and synthetic-data regression tests
- [x] Plan, train and single-fold predict wrappers
- [x] EN/VI documentation, contribution guidelines and CI

## Next

- [ ] Choose a dataset release, access terms, MRI channels and label mapping
- [ ] Add explicit patient-grouped splits and keep an independent test set
- [ ] Train and evaluate a 2D baseline; record configuration and hardware
- [x] Add per-case Dice/IoU with explicit empty-mask handling (desktop: reference mask comparison)
- [ ] Compare 3D configuration when resources permit
- [x] Render mask surfaces in physical coordinates and measure labeled voxel volume
- [x] Export meshes (STL / PLY / OBJ)
- [x] Add native slice and 3D mask-surface viewing
- [ ] Add inference API and model integration
- [x] Save masks in the original image orientation, export CSV/JSON reports
- [x] Manual mask editing with undo/redo, distance rulers and per-region measurements
- [x] DICOM series import (optional SimpleITK) and softmax-entropy uncertainty map
- [x] PACS-style dark workstation layout (2x2, 1+3, 1x1)

No performance results or completion dates are promised by this roadmap.
