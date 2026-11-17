SHELL := /bin/bash
.DEFAULT_GOAL := help

# Override with: make doctor VENV=/path/to/another/environment
VENV ?= $(HOME)/venvs/ai
PYTHON := $(VENV)/bin/python
export PATH := $(VENV)/bin:$(PATH)

DATASET ?= 501
CONFIG ?= 2d
FOLD ?= 0
DATASET_CONFIG ?= configs/dataset.local.json
MANIFEST ?= configs/cases.local.json
INPUT ?= data/inference
OUTPUT ?= artifacts/predictions
MSD_SOURCE ?= data/source/Task01_BrainTumour
WORKERS ?= 2
EXTRA ?=

.PHONY: help check-venv setup setup-dev init-config doctor test check prepare plan train resume predict import-msd desktop setup-desktop

help:
	@printf '%s\n' \
	  'Brain MRI Segmentation — mặc định dùng ~/venvs/ai' \
	  '' \
	  'make setup        Cài dependency nnU-Net vào venv có sẵn' \
	  'make setup-dev    Cài dependency cho test' \
	  'make init-config  Tạo config local từ example, giữ file đã có' \
	  'make setup-desktop Cài PySide6 vào venv có sẵn' \
	  'make desktop     Mở ứng dụng desktop native' \
	  'make doctor       Kiểm tra nnU-Net, PyTorch và CUDA' \
	  'make test         Chạy test với dữ liệu tổng hợp' \
	  'make check        Kiểm tra test và cú pháp shell' \
	  'make import-msd   Nhập dataset MSD Task01 4D đã tải và giải nén' \
	  'make prepare      Kiểm tra và nhập dataset từ config local' \
	  'make plan         Plan và preprocess dataset' \
	  'make train        Train một fold đầy đủ (không phải smoke test)' \
	  'make resume       Tiếp tục checkpoint đang train' \
	  'make predict      Predict bằng fold đã train' \
	  '' \
	  'Ví dụ: make train DATASET=501 CONFIG=3d_fullres FOLD=0' \
	  '        make predict INPUT=data/inference OUTPUT=artifacts/predictions' \
	  '        make train EXTRA="-device cpu"' \
	  '        make doctor VENV=/path/to/venv' \
	  '' \
	  'Cần dataset thật trước prepare/plan/train; chưa có pretrained weights.'

check-venv:
	@test -x "$(PYTHON)" || { printf 'Không tìm thấy Python tại %s. Truyền VENV=/path/to/venv.\n' "$(PYTHON)" >&2; exit 1; }

setup: check-venv
	uv pip install --python "$(PYTHON)" -r requirements.txt

setup-dev: check-venv
	uv pip install --python "$(PYTHON)" -r requirements-dev.txt

init-config:
	@if [ ! -e "$(DATASET_CONFIG)" ]; then mkdir -p "$$(dirname -- "$(DATASET_CONFIG)")"; cp configs/dataset.example.json "$(DATASET_CONFIG)"; fi
	@if [ ! -e "$(MANIFEST)" ]; then mkdir -p "$$(dirname -- "$(MANIFEST)")"; cp configs/cases.example.json "$(MANIFEST)"; fi
	@printf 'Sửa %s và %s theo dữ liệu thật trước khi chạy make prepare.\n' "$(DATASET_CONFIG)" "$(MANIFEST)"

doctor: check-venv
	"$(PYTHON)" scripts/check_environment.py

test: check-venv
	"$(PYTHON)" -m pytest -q

check: test
	bash -n scripts/env.sh scripts/pipeline.sh

prepare: check-venv
	"$(PYTHON)" scripts/prepare_dataset.py --config "$(DATASET_CONFIG)" --manifest "$(MANIFEST)"

plan: check-venv
	bash scripts/pipeline.sh plan "$(DATASET)" "$(CONFIG)"

train: check-venv
	bash scripts/pipeline.sh train "$(DATASET)" "$(CONFIG)" "$(FOLD)" --npz $(EXTRA)

resume: check-venv
	bash scripts/pipeline.sh train "$(DATASET)" "$(CONFIG)" "$(FOLD)" --npz --c $(EXTRA)

predict: check-venv
	bash scripts/pipeline.sh predict "$(DATASET)" "$(CONFIG)" "$(FOLD)" "$(INPUT)" "$(OUTPUT)" $(EXTRA)

import-msd: check-venv
	"$(PYTHON)" scripts/import_msd.py --source "$(MSD_SOURCE)" --dataset-id "$(DATASET)" --workers "$(WORKERS)"

setup-desktop: check-venv
	uv pip install --python "$(PYTHON)" -r requirements-desktop.txt

desktop: check-venv
	"$(PYTHON)" -m desktop $(EXTRA)

.PHONY: kaggle-quota kaggle-build kaggle-prepare kaggle-train kaggle-status kaggle-output
kaggle-quota:
	kaggle quota

kaggle-build: check-venv
	"$(PYTHON)" kaggle/build_notebooks.py

kaggle-prepare:
	kaggle kernels push -p kaggle/prepare -t 14400

kaggle-train:
	kaggle kernels push -p kaggle/train -t 10800

kaggle-status:
	kaggle kernels status hichandut/brain-mri-prepare-cpu
	kaggle kernels status hichandut/brain-mri-2d-100ep

kaggle-output:
	kaggle kernels output hichandut/brain-mri-2d-100ep -p artifacts/kaggle-output
