# Selected dataset / Dataset đã chọn — Liver Tumour (CT)

**Medical Segmentation Decathlon, Task03_Liver**, release `1.0 04/05/2018`.
Archive metadata was checked directly inside the official MONAI-linked archive
(download URL and MD5 confirmed against `monai/apps/datasets.py`; the first
~20MB of the archive was fetched with an HTTP Range request and its
`dataset.json` read directly, rather than guessed).
This is the dataset selected for this project's second task, not a claim of
trained-model performance — no checkpoint has been trained for this task yet.

## Why this dataset?

- Public downloadable CT volumes, masks and a documented channel order.
- 131 labeled training cases — fewer than brain tumour's 484, so expect a
  slower-converging, lower-confidence model until more data or augmentation
  is added.
- Single CT channel; simpler input than brain tumour's four MRI sequences.
- Data metadata specifies **CC-BY-SA 4.0**, separate from the project's AGPL
  code license.

Sources: [challenge](https://medicaldecathlon.com/),
[dataset paper](https://doi.org/10.1038/s41467-022-30695-9),
[MONAI download URL and MD5](https://github.com/Project-MONAI/MONAI/blob/dev/monai/apps/datasets.py).

## Input and labels

MSD stores this task as single-channel 3D CT volumes (no 4D splitting needed, unlike brain tumour).

| Channel | Sequence | Meaning |
|---|---|---|
| 0000 | CT | Portal venous phase CT |

| Label | Name | Tiếng Việt |
|---|---|---|
| 0 | background | Nền / ngoài gan |
| 1 | liver | Gan |
| 2 | cancer | Khối u/tổn thương ác tính trong gan |

These two non-background labels come directly from the official archive's
`dataset.json` (`{"0": "background", "1": "liver", "2": "cancer"}`) — do not
rename or remap them. The selected convention is recorded separately in
`tasks/liver_tumour_ct/dataset.json`; the importer (`scripts/import_msd.py`)
verifies the extracted archive matches this exact fingerprint before
converting it.

## Download and import

**The archive is 28,925,891,584 bytes (~28.9 GB)** — roughly 3.8× the brain
tumour archive (7.6 GB). This is large enough to matter for both local disk
and Kaggle's CPU notebook working disk (commonly ~20GB free beyond the base
image). Extracting only `imagesTr`/`labelsTr`/`dataset.json` (skipping the
70 unlabeled test cases) reduces this somewhat, but CT abdominal volumes can
still be larger per-case than brain MRI. **This has not been fully downloaded
or preprocessed yet in this project** — treat the Kaggle prepare notebook as
unverified until it has actually completed a run; if it fails on disk space,
do not silently reduce the case count — stop and reassess (e.g. attaching a
prebuilt Kaggle Dataset instead of downloading fresh inside the notebook).

```bash
mkdir -p data/downloads data/source
curl -fL -C - \
  https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar \
  -o data/downloads/Task03_Liver.tar
printf '%s\n' 'a90ec6c4aa7f6a3d087205e23d4e6397  data/downloads/Task03_Liver.tar' | md5sum -c -
tar -xf data/downloads/Task03_Liver.tar -C data/source
make import-msd TASK=liver_tumour_ct MSD_SOURCE=data/source/Task03_Liver DATASET=502
make plan TASK=liver_tumour_ct DATASET=502 CONFIG=3d_lowres
make train TASK=liver_tumour_ct DATASET=502 CONFIG=3d_lowres FOLD=0
```

Use `make import-msd TASK=liver_tumour_ct MSD_SOURCE=/path/to/Task03_Liver` for an existing extraction.
Conversion writes `Dataset502_Liver` to `nnUNet_raw`. It refuses an already occupied dataset ID.
The dataset, local paths, models and generated outputs remain outside Git.

## Why 3d_lowres instead of 2d

Brain tumour uses nnU-Net's **2d** configuration (lighter, faster on Kaggle's
free GPU quota). Liver tumour uses **3d_lowres** instead: liver and the
tumours inside it are 3D structures that are easier to separate with
volumetric context, and nnU-Net's own benchmark guidance favors a 3D
configuration for this task family. This makes training heavier per epoch
than the brain tumour 2D pilot — budget Kaggle GPU time accordingly, and
expect a 100-epoch run to need more total sessions to resume through.

## Evaluation plan

Start with **five-fold cross-validation on the 131 labeled cases**, using
nnU-Net's case-level split, same as brain tumour. First run fold 0 to
validate the pipeline, then folds 1–4. A single fold result is preliminary.

Report per-case Dice/IoU for the `liver` and `cancer` labels, with explicit
empty-region handling (`cancer` is absent in some cases). Use out-of-fold
predictions for the overall report. Do not select hyperparameters or
postprocessing on a claimed independent test set.

The **70 official test images have no public masks** in this archive: use
them for inference/demo only, not locally scored accuracy.

**Independent test set.** Unlike brain tumour, no MSD cases were held out of
training: all 131 labeled cases are in the five folds. The scored held-out test
is instead **3D-IRCADb-01** (20 patients, CC BY-NC-ND 4.0), which is unrelated to
MSD/LiTS. Build it with `python scripts/import_ircadb.py`; it lands in
`data/test_sets/liver_tumour_ct/` (see the README there). Its scanners and
contrast phases differ from the training data, so expect a lower Dice than
cross-validation. Use it only for the final report, never for model selection.

With only 131 training cases — fewer than most MSD tasks — and a tumour
label that is visually small and variable, expect published community
benchmarks for this specific task to be lower than for the `liver` label
itself. Do not assume this task will reach brain tumour's reported Dice
range; no training has been run yet to report an actual number.

## Quyết định cho dự án

Dùng 1 kênh CT và 2 nhãn (gan, khối u) ở trên; baseline cho task này là
nnU-Net **3d_lowres** (khác 2D của não — xem lý do ở trên). Đây là task
`liver_tumour_ct` trong `tasks/`, model phân đoạn vùng cơ thể thứ hai sau
`brain_tumour_mri`. Chưa có checkpoint nào được train cho task này —
tài liệu này mô tả dataset đã chọn, không phải kết quả model đã có.
