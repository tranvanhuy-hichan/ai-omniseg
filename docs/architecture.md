# Architecture

The application owns its data manifest, import checks, workflow wrappers and future evaluation/viewer code.
nnU-Net owns preprocessing, planning, neural-network training and inference; it is installed as a pinned dependency.

## Data flow

1. A local manifest identifies registered NIfTI channels and a mask per case.
2. The importer validates all cases before writing a dataset. It copies files without resampling/remapping.
3. nnU-Net checks dataset integrity, generates plans and preprocesses the selected configuration.
4. Training writes fold-specific checkpoints under `nnUNet_results`.
5. Prediction explicitly selects the requested fold and emits masks.

## Current boundaries

Importer checks shape, affine, finite intensities, allowed label values, channel count and duplicate case IDs.
It does not detect patient identity, remove identifying metadata, perform co-registration or create grouped splits.
Multiple visits of the same patient must be handled explicitly by the user.

A native PySide6 slice/mask viewer is available in `desktop/`. No API, model inference, automated diagnostic report or pretrained weights are included.
Future 3D export must preserve the NIfTI voxel-to-world affine, rather than treating voxel indices as millimeters.
