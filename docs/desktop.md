# OmniSeg desktop workspace

The viewer uses **PySide6 / Qt Widgets**. It opens real NIfTI volumes locally;
there is no browser window, server, cloud upload or synthetic diagnosis. The **Mô hình**
dropdown in the header lists every trained model found under `tasks/` (see
[project-structure.md](project-structure.md)); running inference uses whichever one is selected.

```bash
make setup-desktop
make desktop
# Or open a specific volume (and optionally a mask, or start in the 3D view):
make desktop EXTRA="--file path/to/case.nii.gz --view 3d"
```

The Makefile uses `~/venvs/ai` by default. Override with `VENV=/path/to/venv`.

## Available

- Case list from `data/test_sets/<model>/images` when that folder exists, or a folder you select.
- 3D and 4D NIfTI loading in a worker thread.
- Axial, coronal and sagittal voxel planes, slice sliders and mouse-wheel scrolling.
- Channel selection (per the selected model's input order), percentile-based window reset, manual window/level.
- Physical pixel aspect ratio, canonical RAS reorientation and neurological display convention.
- Separate mask loading with strict shape/affine checks and a check against the selected model's label ids.
- In-app segmentation: the **Chạy phân đoạn** button runs `nnUNetv2_predict` for the selected model (fold 0,
  `checkpoint_final.pth`) and shows the result as an AI prediction. It is locked, with the reason shown, when the
  model's checkpoint is missing or the open image has the wrong number of channels.
- A 3D view of the mask surfaces over a faint rendering of the image (a bone preset for CT).
- Colored mask overlay, opacity control, voxel counts and physical volumes when spatial units are known.

Open an image first (folder icon next to the search box, or **Ctrl+O**). To see ground truth, use **Nạp mask** and select the identically named file in `labelsTr`.
The source of the mask is always displayed; a reference mask is not a model prediction.

## Limits

This is a research viewer, not a diagnostic workstation. Oblique scans are displayed on their canonical voxel
planes, not re-sliced into exact anatomical planes; orientation annotations are approximate for those scans.
Geometry checks do not prove that a mask belongs to the same patient or that anatomical registration is correct.
The selected MSD channel names apply only to that known four-channel order; generic NIfTI does not guarantee it.

DICOM support and crosshair navigation are not implemented. Checkpoints are not distributed with the project:
place them under `artifacts/nnUNet_results/` (see the README) before the segmentation button can run a model.

The CPU-only CI tests cover volume processing without importing Qt. Desktop GUI smoke checks require PySide6;
`QT_QPA_PLATFORM=offscreen` can be used on machines without a display.

## Licensing

PySide6/Qt has its own licensing terms; consult the [Qt for Python licenses](https://doc.qt.io/qtforpython-6/licenses.html)
when distributing a packaged application. It is not relicensed by this project's AGPL license.

On Linux, the app uses X11/XWayland for native VTK and its own Fusion theme. It never loads the GTK theme plugin
(avoiding optional GTK/GSettings incompatibilities); when an xdg-desktop-portal service is running it uses the portal
theme so file dialogs are the system's own, otherwise the generic Qt theme. An explicitly set
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

The image volume stays visible while the camera is dragged or zoomed, and the mask surfaces stay readable: depth peeling is kept on at all times so the translucent volume and the translucent mask surfaces are interleaved correctly. The image is a faint context layer (default opacity 12%): MRI is lightly smoothed, and CT uses a fixed Hounsfield window that shows bone and a faint skin shell while leaving soft tissue and air transparent. Peeling costs frame time on integrated GPUs (roughly 15–30 fps on an Intel UHD with a 512×512×129 CT); the "Hiện toàn bộ ảnh" switch turns the volume off for a much faster, mask-only view.
The selected MRI channel is used for the volume. Raw images remain unchanged.
