# Contributing / Đóng góp

Bug reports, documentation, translations and focused code changes are welcome.
For larger features, discuss the intended behavior in an issue first.

1. Fork the repository and branch from `main`.
2. Use Python 3.11 and install `requirements-dev.txt` in a virtual environment.
3. Keep changes focused; test behavior changes using synthetic data.
4. Run `python -m pytest -q` and `bash -n scripts/env.sh scripts/pipeline.sh`.
5. Open a PR describing the problem, the change and its validation.

Full training additionally needs `requirements.txt` and an appropriate PyTorch installation.
Do not commit datasets, patient records, checkpoints, credentials or local manifests.
Do not claim a model improvement without documenting the dataset version, patient split, configuration,
metrics and limitations. Do not select settings using the held-out test set.

Keep English and Vietnamese README instructions consistent when changing setup or commands.
Contributions are licensed under AGPL-3.0-or-later. Follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
Report vulnerabilities through [SECURITY.md](SECURITY.md).
