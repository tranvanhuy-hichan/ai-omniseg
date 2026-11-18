# Selected dataset / Dataset đã chọn

**Medical Segmentation Decathlon, Task01_BrainTumour**, release `2.0 04/05/2018`.
Task-specific metadata was checked directly inside the official MONAI-linked archive.
Machine-readable provenance and evaluation regions: [`configs/msd-brain-tumour-source.json`](../configs/msd-brain-tumour-source.json).
This is the dataset selected for this project's first baseline, not a claim of trained-model performance.

## Why this dataset?

- Public downloadable MRI volumes, masks and a documented channel order.
- 484 labeled training cases, suitable for case-level cross-validation.
- Four registered MRI sequences per case; useful for segmentation and a later 3D viewer.
- Data metadata specifies **CC-BY-SA 4.0**, separate from the project's AGPL code license.

Sources: [challenge](https://medicaldecathlon.com/),
[dataset paper](https://doi.org/10.1038/s41467-022-30695-9),
[MONAI download URL and MD5](https://github.com/Project-MONAI/MONAI/blob/dev/monai/apps/datasets.py).

## Input and labels

MSD stores the four channels in one **4D NIfTI**. Use `make import-msd` to split them with the
upstream nnU-Net converter; do not feed that 4D image directly to the generic 3D manifest importer.

| Channel | Sequence | Meaning |
|---|---|---|
| 0000 | FLAIR | Fluid-attenuated inversion recovery |
| 0001 | T1w | T1-weighted |
| 0002 | t1gd | T1 with gadolinium contrast |
| 0003 | T2w | T2-weighted |

| Label | Name | Tiếng Việt |
|---|---|---|
| 0 | background | Nền / ngoài vùng tổn thương được gán nhãn |
| 1 | edema | Phù nề |
| 2 | non-enhancing tumor | Vùng u không tăng tương phản |
| 3 | enhancing tumour | Vùng u tăng tương phản |

Keep these four mutually exclusive labels for the first model. Do **not** use the earlier binary example
for this dataset and do not assume BraTS labels 1/2/4. The selected convention is recorded separately in
`tasks/brain_tumour_mri/dataset.json`; the importer uses the real source metadata.

For region metrics, derive these unions from the prediction and ground truth:

- **WT / whole tumor region**: labels 1, 2 and 3 (including edema).
- **TC / tumor core**: labels 2 and 3.
- **ET / enhancing tumor**: label 3.

## Download and import

The archive is **7,608,266,240 bytes (~7.6 GB)**. Extracted source, converted data and preprocessed caches
need additional space. The file has not been downloaded in full by choosing this configuration.

```bash
mkdir -p data/downloads data/source
curl -fL -C - \
  https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar \
  -o data/downloads/Task01_BrainTumour.tar
printf '%s\n' '240a19d752f0d9e9101544901065d872  data/downloads/Task01_BrainTumour.tar' | md5sum -c -
tar -xf data/downloads/Task01_BrainTumour.tar -C data/source
make import-msd
make plan DATASET=501 CONFIG=2d
make train DATASET=501 CONFIG=2d FOLD=0
```

Use `make import-msd MSD_SOURCE=/path/to/Task01_BrainTumour` for an existing extraction.
Conversion writes `Dataset501_BrainTumour` to `nnUNet_raw`. It refuses an already occupied dataset ID.
The dataset, local paths, models and generated outputs remain outside Git.

## Evaluation plan

Start with **five-fold cross-validation on the 484 labeled cases**, using nnU-Net's case-level split.
First run fold 0 to validate the pipeline, then folds 1–4. Inspect the generated `splits_final.json`;
if identity information reveals repeat scans, group them by patient before any full training.
A single fold result is preliminary. Do not split individual slices randomly across folds.

Report per-case Dice/IoU for labels and the WT/TC/ET regions, with explicit empty-region handling.
Use out-of-fold predictions for the overall report. Do not select hyperparameters or postprocessing on
a claimed independent test set. Full cross-validation evaluation code has not been implemented yet.

The **266 official test images have no public masks** in this archive: use them for inference/demo only,
not locally scored accuracy. There is no independent labeled test set in the initial plan, so no claim
of external generalization or healthy-vs-tumor diagnosis should be made.

## Quyết định cho dự án

Dùng 4 kênh MRI và 4 nhãn ở trên; baseline đầu tiên là nnU-Net **2D**, sau đó thử 3D khi đủ VRAM.
Giữ nhãn nhiều lớp để app OmniSeg sau này bật/tắt từng vùng và dựng 3D. Đây là task `brain_tumour_mri`
trong `tasks/`; các model phân đoạn vùng cơ thể khác (phổi, tim…) sẽ là các task riêng cùng cấu trúc.
Đây là bài toán **phân đoạn vùng tổn thương**, không phải phân loại mọi bệnh não hoặc khẳng định chẩn đoán.
