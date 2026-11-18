# Project structure / Cấu trúc thư mục

This maps what each top-level folder is for. For the end-to-end data flow
(importer → nnU-Net → mask → viewer), see [architecture.md](architecture.md).

```text
.
├── omniseg/                 OmniSeg: native multi-model viewer (PySide6 + VTK)
│   ├── app.py                 Qt main window: slice panels, inspector, model picker, in-app inference
│   ├── registry.py            Scans tasks/ for task.json files, no Qt dependency
│   ├── core/                  Non-Qt logic, reusable outside the GUI
│   │   ├── volume.py            NIfTI loading, orientation, slice rendering
│   │   ├── inference.py          Builds nnUNetv2_predict env/command for a given task
│   │   ├── anatomy.py            VTK volume rendering of the raw image intensities
│   │   └── surfaces.py           Mask label surfaces (marching cubes + decimation)
│   ├── view3d.py                3D surface/volume view embedded in the Qt workspace
│   ├── assets/                  SVG icons (logo, switches, chevrons) used by the Qt stylesheet
│   ├── entry_point.py           PyInstaller entry point (`python -m omniseg` uses __main__.py instead)
│   ├── __main__.py               `python -m omniseg` launcher
│   └── omniseg.spec             PyInstaller build spec, see "Building the desktop app" below
│
├── tasks/                   One folder per trained model (body region/task); see tasks/README.md
│   └── brain_tumour_mri/        task.json (dataset id/config/fold/checkpoint/channels/labels), dataset.json, README.md
│       (adding a new model — e.g. lung CT, heart CT — only needs a new tasks/<id>/ folder, no app code change)
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
├── configs/                 Dataset/case manifest examples and local overrides (*.local.json is gitignored)
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

## Adding a new model (new body region/modality)

OmniSeg is multi-model: each trained checkpoint is a self-contained **task** under
`tasks/<task_id>/`, and `omniseg/registry.py` scans that folder to populate the
app's model dropdown. See [tasks/README.md](../tasks/README.md) for the exact
`task.json` schema and the steps to plug in a new model (e.g. lung CT, heart CT)
without touching `omniseg/` code.

## Building the desktop app

`omniseg/omniseg.spec` is a PyInstaller spec that bundles the Qt viewer, its
VTK rendering code and `omniseg/assets/` into a standalone app — no Python
install required to run it.

```bash
make build-desktop          # writes dist/OmniSeg/ on the current platform
```

`.github/workflows/build-desktop.yml` runs the same build on Linux, Windows
and macOS runners. It triggers on a `v*` tag push (and can be run manually via
`workflow_dispatch`), zips each platform's `dist/OmniSeg/` folder, uploads it
as a workflow artifact, and attaches it to the GitHub Release created for that
tag. It does not run on every push — PyInstaller + VTK builds are too heavy to
justify that on ordinary commits; see `.github/workflows/ci.yml` for the tests
that do run on every push.
