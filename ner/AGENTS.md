# Repository Guidelines

## Project Structure & Module Organization
- Core scripts live in `ner/`: training (`ner_train.py`), synthetic data generation (`build_standard_dataset.py`), fuzzy-labeled real data conversion (`build_real_dataset.py`), dataset merging (`merge_datasets.py`), and Hub upload helper (`hugging_face_up.py`).
- Configuration defaults sit in `ner/configs/train_default.json` and assume commands are run from the repository root so paths like `ner/datasets/...` resolve correctly.
- Generated data lands under `ner/datasets/` (e.g., `standard/`, `real/`, `combined/`), and training artifacts are stored in `ner/artifacts/`.

## Build, Test, and Development Commands
- Generate synthetic data (expects administrative sources under `fuzz/address_parser/data`):
  `python ner/build_standard_dataset.py --data-dir fuzz/address_parser/data --output-dir ner/datasets/standard`
- Convert raw address dumps using the fuzzy parser:  
  `python ner/build_real_dataset.py --input <file> --output-dir ner/datasets/real`
- Merge datasets into a single split:  
  `python ner/merge_datasets.py --train-files ner/datasets/standard/train.jsonl ner/datasets/real/train.jsonl --output-dir ner/datasets/combined`
- Train the Electra NER model:  
  `python ner/ner_train.py --config ner/configs/train_default.json`
- Upload artifacts to Hugging Face (requires `HF_TOKEN` in `.env` or env):  
  `python ner/hugging_face_up.py --upload both`

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation; prefer type hints, dataclasses, and small helper functions as in existing scripts.
- Keep CLI entry points using `argparse` and clear docstrings; use snake_case for files, functions, and JSON keys (`tokens`, `ner_tags`, `source`).
- Keep JSONL rows minimal and consistent; preserve UTF-8 text but keep code comments ASCII.

## Testing Guidelines
- There is no formal test suite; validate changes by running the relevant CLI with small samples: generate standard data, merge, then run `ner_train.py` to ensure tokenization and training complete without errors.
- When adding logic, prefer deterministic seeds and print concise split counts so dataset regressions are obvious.

## Commit & Pull Request Guidelines
- Git history is terse; keep commit subjects short and imperative (e.g., `add merged dataset recipe`, `tune electra lr`). Include context in the body when changing data formats or training defaults.
- PRs should list the command(s) executed, datasets touched (paths), and any new artifact locations or metric deltas (precision/recall/F1). Link related issues when available and add a brief summary of manual verification steps.

## Security & Configuration Tips
- Avoid committing raw address dumps or secrets; keep tokens in `.env` or environment variables and rely on the `--token-env` flag when uploading.
- Default paths include the `ner/` prefix; if running from inside the `ner/` folder, adjust arguments or set absolute paths to avoid missing-file errors.
