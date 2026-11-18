# Liver Tumour (CT)

Second task trained in this project. Source dataset, licensing and channel/label
provenance are documented in [docs/dataset-liver.md](../../docs/dataset-liver.md) —
this file only covers what is specific to this task's configuration.

- `task.json` — metadata the app and scripts read: nnU-Net dataset id/config/fold/
  trainer/checkpoint, input channel order, and label ids. This is the single
  place that pins which trained checkpoint OmniSeg runs for this task.
- `dataset.json` — the nnU-Net dataset fingerprint used when this dataset was
  imported (channel names, labels, file ending).

Source: Medical Segmentation Decathlon, Task03_Liver, release `1.0 04/05/2018`.
131 labeled training cases, 70 unlabeled test cases, single-channel CT, two
non-background labels (`liver`, `cancer`). License CC-BY-SA 4.0.

No checkpoint has been trained for this task yet — `tasks/liver_tumour_ct/`
is a scaffold so `omniseg/registry.py` and the Kaggle pipeline can target it
once training (see [docs/kaggle-training.md](../../docs/kaggle-training.md))
produces `artifacts/nnUNet_results/Dataset502_Liver/.../fold_0/checkpoint_final.pth`.
Until that checkpoint exists, selecting this task in the app's model dropdown
will fail inference with a missing-checkpoint error, not a silent wrong answer.
