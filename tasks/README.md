# Tasks

A **task** is one body region / segmentation problem OmniSeg can run: one
trained nnU-Net checkpoint, one input channel layout, one label set. Each task
lives in its own folder here and is the only thing `omniseg/registry.py` reads
to populate the app's "Model" selector — adding a new task never requires
changing app code.

```text
tasks/<task_id>/
├── task.json      required — see schema below
├── dataset.json   nnU-Net dataset fingerprint (channel names, labels, file ending)
└── README.md      data source, license, notes specific to this task
```

## `task.json` schema

| Field | Meaning |
|---|---|
| `id` | Folder name; stable identifier used in configs/CLI. |
| `display_name` | Shown in the OmniSeg model dropdown. |
| `modality` | Imaging modality, e.g. `"MRI"`, `"CT"`. |
| `nnunet_dataset_id` | nnU-Net numeric dataset id (`-d` for `nnUNetv2_*`). |
| `nnunet_dataset_name` | Matching `Dataset<id>_<Name>` folder name. |
| `nnunet_config` | nnU-Net plan config, e.g. `"2d"`, `"3d_fullres"`. |
| `fold` | Trained fold to run inference with. |
| `trainer` | nnU-Net trainer class name used for training. |
| `checkpoint` | Checkpoint filename to load (`-chk`). |
| `channel_names` | Input channel order, as `{"0": "...", ...}`. |
| `labels` | Label name → integer id, including `"background": 0`. |

## Adding a new model (e.g. lung CT, heart CT)

`tasks/liver_tumour_ct/` is a worked example: scaffolded with its `task.json`/
`dataset.json`/`README.md` and a Kaggle prepare/train pipeline, but **no
checkpoint trained yet** (see [docs/dataset-liver.md](../docs/dataset-liver.md)
for the dataset it targets). Use it as a template for the next task.

1. Train the nnU-Net model as usual (see [docs/kaggle-training.md](../docs/kaggle-training.md)),
   picking a dataset id that doesn't collide with an existing task (brain tumour
   uses `501`, liver tumour uses `502`).
2. Create `tasks/<new_task_id>/task.json` with that dataset's id/config/fold/
   trainer/checkpoint, channel order and labels.
3. Copy the `dataset.json` fingerprint used for that dataset into the same
   folder, and write a short `README.md` with the data source and license.
4. Make sure the matching `data/nnUNet_raw/Dataset<id>_...` and
   `artifacts/nnUNet_results/Dataset<id>_.../...` exist locally (or wherever
   `nnUNet_raw`/`nnUNet_results` point).

No change to `omniseg/` is needed — the app discovers the new task from its
`task.json` the next time it starts.
