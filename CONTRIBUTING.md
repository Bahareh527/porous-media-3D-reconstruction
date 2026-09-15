# Contributing

This repository is a curated research record. Corrections that improve reproducibility, documentation, or compatibility are welcome.

1. Open an issue describing the proposed change and the notebook or result it affects.
2. Do not commit raw tomographic volumes, trained model weights, credentials, personal absolute paths, or publisher files without redistribution permission.
3. Keep maintained notebooks free of execution output. Do not reintroduce older experimental notebook copies.
4. Install the development extras and run `python -m pytest`, `python -m ruff check src tests scripts`, `python scripts/check_repository.py`, and `python scripts/execute_notebooks.py` before submitting a pull request. Use `scripts/refresh_notebooks.py` deliberately when changing notebook templates; it refreshes maintained checksums.
5. Explain any scientific or numerical change and provide the environment and data subset used to verify it.

Questions about the scientific interpretation or reuse of thesis material should be directed to the author through an appropriate academic channel.
