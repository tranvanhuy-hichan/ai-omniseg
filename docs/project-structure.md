# Project structure / Cấu trúc thư mục

This maps what each top-level folder is for. For the end-to-end data flow
(importer → nnU-Net → mask → viewer), see [architecture.md](architecture.md).

```text
.
├── desktop/                 Native NeuroSeg viewer (PySide6 + VTK)
│   ├── app.py                 Qt main window: slice panels, inspector, in-app inference
│   ├── volume.py               NIfTI loading, orientation, slice rendering (no Qt)
│   ├── anatomy.py              VTK volume rendering of the raw MRI intensities
│   ├── surfaces.py             Mask label surfaces (marching cubes + decimation)
│   ├── view3d.py                3D surface/volume view embedded in the Qt workspace
│   ├── assets/                  SVG icons (logo, switches, chevrons) used by the Qt stylesheet
│   ├── entry_point.py           PyInstaller entry point (`python -m desktop` uses __main__.py instead)
│   ├── __main__.py               `python -m desktop` launcher
│   └── neuroseg.spec            PyInstaller build spec, see "Building the desktop app" below
│
├── scripts/                 Shell/CLI glue around nnU-Net, independent of the desktop app
│   ├── env.sh                   Exports nnUNet_raw/preprocessed/results defaults
│   ├── pipeline.sh               plan / train / predict wrapper around nnUNetv2_* commands
│   ├── import_msd.py             Verifies and converts the MSD Task01_BrainTumour archive
│   ├── prepare_dataset.py         Validates a custom dataset manifest before import
│   └── check_environment.py       Reports installed nnU-Net/PyTorch/CUDA versions
│
├── kaggle/                  Notebook sources for the two-stage Kaggle training pilot
│   ├── prepare/                  CPU notebook: download, verify, preprocess, 5-fold split
│   ├── train/                     GPU notebook: time-boxed nnU-Net training, resumable
│   ├── prepare_cpu.py             Source written into the prepare notebook
│   ├── train_budget.py            Source written into the train notebook (epoch-boundary budget)
│   ├── build_notebooks.py         Regenerates notebook JSON from the .py sources above
│   └── launch_after_prepare.py    Starts the train notebook once prepare finishes
│
├── configs/                 Dataset/case manifests (local *.local.json is gitignored)
├── tests/                   pytest suite; synthetic data only, no patient data or checkpoints
├── docs/                    This file, architecture, dataset provenance, desktop/Kaggle usage
├── .github/workflows/       CI (tests) and desktop app builds, see below
│
├── Makefile                 `make <target>` wrappers; see `make help`
├── requirements*.txt        Runtime / dev / desktop-only dependency pins
└── README.md / README.vi.md Project overview (English / Vietnamese)
```

Not tracked in git (created locally or downloaded, see `.gitignore`):
`data/` (datasets), `artifacts/` (checkpoints, predictions), `.venv/`.

## Building the desktop app

`desktop/neuroseg.spec` is a PyInstaller spec that bundles the Qt viewer, its
VTK rendering code and `desktop/assets/` into a standalone app — no Python
install required to run it.

```bash
make build-desktop          # writes dist/NeuroSeg/ on the current platform
```

`.github/workflows/build-desktop.yml` runs the same build on Linux, Windows
and macOS runners. It triggers on a `v*` tag push (and can be run manually via
`workflow_dispatch`), zips each platform's `dist/NeuroSeg/` folder, uploads it
as a workflow artifact, and attaches it to the GitHub Release created for that
tag. It does not run on every push — PyInstaller + VTK builds are too heavy to
justify that on ordinary commits; see `.github/workflows/ci.yml` for the tests
that do run on every push.
