# Brain Tumour (MRI)

First task trained in this project. Source dataset, licensing and channel/label
provenance are documented in [docs/dataset.md](../../docs/dataset.md) — this file
only covers what is specific to this task's configuration.

- `task.json` — metadata the app and scripts read: nnU-Net dataset id/config/fold/
  trainer/checkpoint, input channel order, and label ids. This is the single
  place that pins which trained checkpoint OmniSeg runs for this task.
- `dataset.json` — the nnU-Net dataset fingerprint used when this dataset was
  imported (channel names, labels, file ending).

Pilot checkpoint: fold 0, trained the full 100/100 epochs. Not validated enough
to use for diagnosis — see the in-app disclaimer.
