# OmniSeg desktop workspace

The viewer uses **PySide6 / Qt Widgets**. It opens real NIfTI volumes locally;
there is no browser window, server, cloud upload or synthetic diagnosis. The **Mô hình**
dropdown in the header lists every trained model found under `tasks/` (see
[project-structure.md](project-structure.md)); running inference uses whichever one is selected.

```bash
make setup-desktop
make desktop
# Or open a specific volume:
make desktop EXTRA="--file data/source/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz"
```

The Makefile uses `~/venvs/ai` by default. Override with `VENV=/path/to/venv`.

## Available

- Case list from the downloaded MSD `imagesTr` folder, or a folder you select.
- 3D and 4D NIfTI loading in a worker thread.
- Axial, coronal and sagittal voxel planes, slice sliders and mouse-wheel scrolling.
- Channel selection (per the selected model's input order), percentile-based window reset, manual window/level.
- Physical pixel aspect ratio, canonical RAS reorientation and neurological display convention.
- Separate mask loading with strict shape/affine and label 0–3 checks.
- Colored mask overlay, opacity control, voxel counts and physical volumes when spatial units are known.

Open an image first (folder icon next to the search box, or **Ctrl+O**). To see ground truth, use **Nạp mask** and select the identically named file in `labelsTr`.
The source of the mask is always displayed; a reference mask is not a model prediction.

## Limits

This is a research viewer, not a diagnostic workstation. Oblique scans are displayed on their canonical voxel
planes, not re-sliced into exact anatomical planes; orientation annotations are approximate for those scans.
Geometry checks do not prove that a mask belongs to the same patient or that anatomical registration is correct.
The selected MSD channel names apply only to that known four-channel order; generic NIfTI does not guarantee it.

Model inference, DICOM support, crosshair navigation and packaging are not yet implemented.
The segmentation button is disabled until inference integration is implemented; training a model alone does not enable it.

The CPU-only CI tests cover volume processing without importing Qt. Desktop GUI smoke checks require PySide6;
`QT_QPA_PLATFORM=offscreen` can be used on machines without a display.

## Licensing

PySide6/Qt has its own licensing terms; consult the [Qt for Python licenses](https://doc.qt.io/qtforpython-6/licenses.html)
when distributing a packaged application. It is not relicensed by this project's AGPL license.

On Linux, the app uses X11/XWayland for native VTK and its own Fusion theme and defaults to the generic Qt platform theme,
avoiding optional GTK/GSettings theme-plugin incompatibilities. An explicitly set
`QT_QPA_PLATFORMTHEME` is preserved.

## Bố cục và phóng lớn mặt phẳng

- Bố cục mặc định: Axial lớn bên trái, Coronal/Sagittal bên phải; thống kê ở thanh điều khiển có thể cuộn.
- Bấm **Phóng lớn** ở góc ô ảnh hoặc nhấp đúp ảnh để xem một mặt phẳng trên toàn vùng làm việc.
- Ô được chọn thay thế toàn bộ vùng chứa ba ảnh; hai thanh bên vẫn dùng được. Thanh đầu được gộp thành một hàng tên ca, chuỗi MRI và bố cục.
- Bấm **Thu lại**, nhấp đúp lần nữa, hoặc nhấn **Esc** để khôi phục bố cục và giữ nguyên vị trí lát cắt.
- Hộp **Bố cục** cho phép chuyển trực tiếp giữa ba mặt phẳng và chế độ nhiều ô.

This expands a viewport within the application; it does not resize or alter the source MRI data.
Optional Qt regression tests run headlessly when PySide6 is installed; otherwise they are skipped.

## 3D surfaces from a mask

1. Open an image, then **Nạp mask** and choose the same case's mask.
2. Select **3D · Não và mask** from **Bố cục**.
3. Drag with the left mouse button to rotate, use the middle button to pan, and scroll to zoom.
4. Use the right inspector to toggle **Hiện toàn bộ MRI não**, each mask region, separate opacity sliders, and anatomical camera directions.
5. Press **Esc** or choose another layout to return to slices.

For a local sample:

```bash
make desktop EXTRA="--file data/source/Task01_BrainTumour/imagesTr/BRATS_102.nii.gz --mask data/source/Task01_BrainTumour/labelsTr/BRATS_102.nii.gz --view 3d"
```

VTK builds separate surfaces for MSD labels 1, 2 and 3 in a worker thread. Background padding closes regions
that touch the volume boundary. Surface vertices are transformed using the complete NIfTI affine (rotation,
translation, scaling and reflection), rather than assuming unit-spaced voxels. Coordinates are the image's
RAS world coordinates; units come from the source header. The orientation cube indicates R/L, A/P and S/I.

Surfaces interpolate the boundary halfway between voxel centers; no mesh smoothing or editing is applied.
The volume statistics remain counts of labeled voxels times the voxel volume, not mesh-derived measurements.
An empty mask or absent label produces no surface for that label. Changing the case or clearing a mask
invalidates pending results so an older case's surface cannot replace the new one.

This mode renders MRI intensity as a grayscale volume and the mask as separate colored surfaces.
The **Hiện toàn bộ MRI não** checkbox can show/hide the MRI independently of the mask.
For the skull-stripped MSD data, this provides brain context; arbitrary MRI files may include scalp/skull or other anatomy.
This is intensity rendering, not an anatomical brain segmentation. Empty MRI background is cropped with its
voxel offset preserved so it remains aligned to mask surfaces. The viewer preserves mask provenance and does not run AI inference. VTK/Qt's native interactor uses X11/XWayland on Linux;
the desktop entry point defaults to `QT_QPA_PLATFORM=xcb` unless an explicit platform was supplied.

3D controls are in the right inspector; no separate 3D title/source toolbar consumes image space.
The selected MRI channel is used for the volume. Raw images remain unchanged.
