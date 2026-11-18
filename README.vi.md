# NeuroSeg

**Tiếng Việt** · [English](README.md)

**NeuroSeg** là dự án độc lập dùng **nnU-Net v2 như dependency**, không fork hoặc sao chép framework.
Mục tiêu ban đầu: phân đoạn vùng u não trên MRI, đánh giá theo bệnh nhân và tạo bề mặt 3D từ mask.
Đây là dự án nghiên cứu, chưa phải công cụ chẩn đoán lâm sàng.

![Viewer desktop NeuroSeg hiển thị mask khối u do AI dự đoán](docs/images/desktop-viewer-segmentation.png)

## Trạng thái

- Đã có importer NIfTI nhiều kênh, kiểm tra geometry và nhãn trước khi tạo dataset.
- Đã có lệnh plan/preprocess, train theo fold và predict theo đúng fold đã train.
- Repo không phân phối dataset, checkpoint hoặc kết quả đánh giá; dataset đã tải chỉ nằm trên máy cục bộ.
- Đã có viewer desktop xem lát cắt và mask; đã có dựng bề mặt 3D từ mask; đã chạy được checkpoint đang huấn luyện ngay trong viewer.

## Cài đặt

Dùng Python 3.11 và môi trường riêng. Với `uv`:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install --torch-backend auto -r requirements.txt
uv pip install -r requirements-dev.txt
```

Nếu dùng pip, cài bản PyTorch phù hợp GPU/CPU theo hướng dẫn chính thức trước, sau đó:

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
```

`requirements.txt` ghim nnU-Net 2.8.1. PyTorch/CUDA được chọn theo hệ điều hành và driver;
không coi môi trường CUDA trên máy này là cấu hình dùng được cho mọi máy.

## Chạy nhanh bằng Makefile

Mặc định dùng `~/venvs/ai`, không cần activate và không tạo venv mới.
Có thể đổi bằng `make doctor VENV=/duong/dan/venv`.

```bash
make help
make setup          # cài nnU-Net vào venv có sẵn (cần uv)
make doctor
make check
make init-config    # sau đó sửa hai file configs/*.local.json theo dataset thật
make prepare
make plan
make train          # dataset 501, cấu hình 2d, fold 0; train đầy đủ
make predict
```

Chỉ chạy prepare/plan/train khi đã có dataset và mask đúng quy ước.
`make setup-dev` cài dependency test theo requirements-dev.txt, có thể điều chỉnh
phiên bản package trong môi trường dùng chung; không cần chạy lại nếu test đã chạy được.

```bash
make train CONFIG=3d_fullres FOLD=0
make resume CONFIG=2d FOLD=0
make predict INPUT=data/inference OUTPUT=artifacts/predictions
```

## Chuẩn bị dữ liệu

Đầu vào là các thể tích `.nii.gz` và mask, không phải ảnh chụp màn hình MRI.
Ví dụ cấu hình có bốn kênh **FLAIR, T1, T1ce, T2**, theo đúng thứ tự đó.
Mọi kênh và mask của một ca phải được co-register, có cùng shape và affine.
Importer chỉ kiểm tra geometry; nó không tự đăng ký ảnh hoặc bảo đảm ảnh đã khớp giải phẫu.

1. Đặt dữ liệu trong `data/source/` (được Git bỏ qua).
2. Copy hai file example trong `configs/`, sửa danh sách ca và đường dẫn.
3. Sửa cấu hình nhãn phù hợp bộ dữ liệu. Ví dụ hiện tại là **mask nhị phân**: nền=0, u=1.
   Mask BraTS có thể dùng nhãn khác theo phiên bản; không dùng thẳng mask nhiều lớp với ví dụ nhị phân.
   Importer sẽ từ chối nhãn ngoài cấu hình, không tự gộp hoặc đổi nhãn.

```bash
cp configs/dataset.example.json configs/dataset.local.json
cp configs/cases.example.json configs/cases.local.json
source scripts/env.sh
python scripts/prepare_dataset.py \
  --config configs/dataset.local.json \
  --manifest configs/cases.local.json
```

Đường dẫn trong manifest được tính từ thư mục chứa manifest; đường dẫn tuyệt đối cũng được hỗ trợ.
Mỗi ID phải là một bệnh nhân độc lập. Với nhiều lần chụp của cùng người, cần tự tạo split theo bệnh nhân
để tránh rò rỉ giữa train/validation. Không đưa tập test độc lập vào manifest huấn luyện.

Kết quả:

```text
data/nnUNet_raw/Dataset501_BrainTumor/
├── dataset.json
├── imagesTr/
│   ├── patient001_0000.nii.gz
│   ├── patient001_0001.nii.gz
│   ├── patient001_0002.nii.gz
│   └── patient001_0003.nii.gz
└── labelsTr/
    └── patient001.nii.gz
```

Importer từ chối ghi đè dataset đã tồn tại. Năm fold mặc định cần ít nhất 5 ca;
đây chỉ là số tối thiểu để chia fold, không phải lượng dữ liệu đủ cho kết quả đáng tin cậy.

## Plan, train và predict

Bắt đầu bằng cấu hình 2D để thử pipeline. Lệnh train dưới đây là một lượt huấn luyện đầy đủ,
không phải smoke test vài bước. Máy RTX 3050 4 GB có thể cần cấu hình tài nguyên nhỏ hơn;
chưa đo mức dùng VRAM của dataset thực tế.

```bash
scripts/pipeline.sh plan 501 2d
scripts/pipeline.sh train 501 2d 0 --npz
```

Sau khi train fold 0 xong:

```bash
scripts/pipeline.sh predict 501 2d 0 data/inference artifacts/predictions
```

Input inference phải có đủ kênh với hậu tố `_0000.nii.gz`, `_0001.nii.gz`… giống lúc train.
Muốn tiếp tục checkpoint đang train: thêm `--c` vào lệnh train.
Muốn thử 3D: plan rồi train với `3d_fullres`, sau khi kiểm tra bộ nhớ phù hợp.
Một fold không đại diện cho kết quả cross-validation đủ 5 fold.

`scripts/env.sh` tạo ba đường dẫn nnU-Net trong repo. Có thể export `nnUNet_raw`,
`nnUNet_preprocessed`, `nnUNet_results` trước để dùng ổ đĩa khác.
Dữ liệu, checkpoint và output lớn đều được Git bỏ qua.

## Kiểm tra

```bash
python scripts/check_environment.py
python -m pytest -q
scripts/pipeline.sh --help
```

Test dùng dữ liệu NIfTI tổng hợp để kiểm tra việc nhập dữ liệu và từ chối geometry/nhãn không hợp lệ;
không đo chất lượng mô hình trên dữ liệu thật.

## Cấu trúc thư mục

Xem [project-structure.md](docs/project-structure.md) (tiếng Anh) để biết từng thư mục gốc
(`desktop/`, `scripts/`, `kaggle/`, `configs/`, `tests/`, `docs/`) dùng để làm gì, và cách
đóng gói app desktop thành file chạy độc lập.

## Bước tiếp theo

1. Chọn phiên bản dataset MRI và quy ước nhãn, chuẩn bị split theo bệnh nhân.
2. Train baseline 2D, đánh giá Dice/IoU trên ca giữ riêng; báo cáo lỗi và vùng rỗng rõ ràng.
3. So sánh với cấu hình 3D khi tài nguyên cho phép.
4. Tạo mesh từ mask bằng Marching Cubes với tọa độ vật lý đúng affine; thêm viewer và đo thể tích.

## Nguồn và ghi nhận

- [nnU-Net — mã nguồn và giấy phép Apache-2.0](https://github.com/MIC-DKFZ/nnUNet)
- [Định dạng dataset](https://github.com/MIC-DKFZ/nnUNet/blob/v2.8.1/documentation/dataset_format.md)
- [Pipeline train/inference](https://github.com/MIC-DKFZ/nnUNet/blob/v2.8.1/documentation/how_to_use_nnunet.md)
- [Cài đặt PyTorch](https://pytorch.org/get-started/locally/)
- Isensee et al., *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation*,
  Nature Methods (2021), [DOI](https://doi.org/10.1038/s41592-020-01008-z).

## Open source

Code gốc dùng [AGPL-3.0-or-later](LICENSE), giống Nestor Agent. nnU-Net và các dependency giữ giấy phép riêng.
Xem [hướng dẫn đóng góp](CONTRIBUTING.md), [quy tắc ứng xử](CODE_OF_CONDUCT.md),
[báo lỗi bảo mật](SECURITY.md) và [lộ trình](docs/roadmap.md).
CI chỉ kiểm tra importer bằng dữ liệu tổng hợp, không xác nhận chất lượng y khoa.

## Selected dataset / Dataset đã chọn

The initial baseline uses **MSD Task01 BrainTumour (release 2.0)**, with 4 MRI channels and
labels 0=background, 1=edema, 2=non-enhancing tumor, 3=enhancing tumor.
See [dataset decision, download and evaluation plan](docs/dataset.md).
Use `make import-msd` for its 4D files; the earlier binary manifest example is for custom data only.

## Viewer desktop native

```bash
make setup-desktop
make desktop
```

Xem lát cắt NIfTI thật, các kênh MRI và mask nạp riêng, hoặc chạy trực tiếp checkpoint
đang huấn luyện ngay trong viewer.

Xem [cách dùng và giới hạn của viewer desktop](docs/desktop.md). Đã có dựng thể tích MRI và bề mặt 3D từ mask bằng VTK.
Nút "Chạy phân đoạn" gọi checkpoint thử nghiệm hiện tại (epoch 43/100, fold 0) qua
`nnUNetv2_predict`; kết quả luôn được gắn nhãn là kết quả AI, không bao giờ coi là nhãn tham chiếu.

## Kaggle training

CPU preprocessing and a budgeted 100-epoch 2D pilot are prepared in `kaggle/`.
See [Kaggle stages, quota, checkpoints and commands](docs/kaggle-training.md).
