# Brain MRI Segmentation

**Tiếng Việt** · [English](README.md)

Dự án độc lập dùng **nnU-Net v2 như dependency**, không fork hoặc sao chép framework.
Mục tiêu ban đầu: phân đoạn vùng u não trên MRI, đánh giá theo bệnh nhân và tạo bề mặt 3D từ mask.
Đây là dự án nghiên cứu, chưa phải công cụ chẩn đoán lâm sàng.

## Trạng thái

- Đã có importer NIfTI nhiều kênh, kiểm tra geometry và nhãn trước khi tạo dataset.
- Đã có lệnh plan/preprocess, train theo fold và predict theo đúng fold đã train.
- Chưa có dataset thật, checkpoint, kết quả đánh giá hoặc ứng dụng web.
- Dựng 3D và viewer là bước tiếp theo; không phải tính năng đã hoàn thành.

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
