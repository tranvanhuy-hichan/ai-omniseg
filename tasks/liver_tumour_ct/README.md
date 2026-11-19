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

Training: nnU-Net `3d_lowres`, fold 0 of 5 (104 train / 27 validation cases), `nnUNetTrainer_100epochs`,
100/100 epochs on a Kaggle Tesla T4 (about 6.2 hours). The checkpoint itself is not in the repository: place
`checkpoint_final.pth` (with `plans.json` and `dataset.json`) under
`artifacts/nnUNet_results/Dataset502_Liver/nnUNetTrainer_100epochs__nnUNetPlans__3d_lowres/fold_0/` to enable
the Run button. Without it the app keeps the button locked and shows why, instead of running and failing.
The only metric so far is training-time pseudo-Dice on fold-0 validation patches (liver 0.96, cancer 0.77 at the
last epoch), which is not full-volume Dice. See the README's *Models* section for the full configuration.
