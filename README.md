# Vietnamese Address Parser + NER Toolkit
This repo combines a fuzzy address parser with a full synthetic/real dataset pipeline for training Electra-based Named Entity Recognition models that understand the Vietnamese administrative hierarchy (old 3-level + new 2-level).

## Highlights
- **Hybrid parser** (`inexus/inexus_parser.py`): Fuzzy matching across wards/districts/provinces, aware of legacy→modern mappings.
- **Synthetic dataset builder** (`ner/build_standard_dataset.py`): Generates millions of auto-labeled addresses (Street/Ward/District/Province) with configurable variants (accentless, abbreviations, compact forms, etc.).
- **Real dataset ingester** (`ner/build_real_dataset.py`): Parses raw address dumps (JSON/JSONL), labels street + administrative spans, and supports `--load-mode memory|batch|stream` so you can toggle between full RAM, chunked batches, or low-RAM streaming.
- **Dataset merger** (`ner/merge_datasets.py`): Concatenates any number of JSONL splits and re-splits with deterministic shuffling.
- **Electra trainer** (`ner/ner_train.py`): Fine-tunes `NlpHUST/electra-base-vn` (or any HF checkpoint) using a JSON config instead of a huge CLI.

## Directory Structure
- `inexus/`: Address parser + `data/` (administrative dumps & mappings).
- `ner/`:
  - `build_standard_dataset.py`: synthetic data generator.
  - `build_real_dataset.py`: real address labeling.
  - `merge_datasets.py`: dataset blender.
  - `configs/`: JSON configs for training.
  - `ner_train.py`: Hugging Face Trainer wrapper.
- `tests/`: Parser/unit tests.

## Quick Start
### 1. Install deps
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 1b. Optional: Download public dataset
Grab the curated samples from Kaggle: [NER Address Standard Dataset](https://www.kaggle.com/datasets/thnhthunh/ner-address-standard-dataset) and drop the downloaded files under `ner/datasets/` (the repo ignores the root-level `datasets/` folder by default).

### 2. Generate synthetic data
```bash
python ner/build_standard_dataset.py --max-samples 500000
# Outputs ner/datasets/{train,test}.jsonl with `source`=old/new metadata.
```

### 3. Label real addresses (optional but recommended)
```bash
python ner/build_real_dataset.py \
  --address-file ner/datasets/addresses.jsonl \
  --output-dir ner/datasets/real \
  --load-mode batch \
  --batch-size 20000   # use memory for full load, stream for ultra-low RAM
```

### 4. Merge sources & split
```bash
python ner/merge_datasets.py \
  --train-files ner/datasets/standard/train.jsonl ner/datasets/real/train_real.jsonl \
  --test-files ner/datasets/standard/test.jsonl ner/datasets/real/test_real.jsonl \
  --output-dir ner/datasets/combined
```

### 5. Train Electra NER
Edit `ner/configs/train_default.json` (update `train_file`, `eval_file`, hyperparams) then run:
```bash
python ner/ner_train.py --config ner/configs/train_default.json
```
The Trainer will log to `ner/artifacts/`, save the best checkpoint, and print precision/recall/F1 via `seqeval`.
> TPU/Kaggle note: `ner_train.py` automatically switches to the non-fused `adamw_hf` optimizer whenever `torch-xla` is detected. Override `optim` inside the JSON config if you need a specific optimizer.

## Configuration Tips
- `build_standard_dataset.py` accepts `--max-samples`, `--train-ratio`, etc., so you can produce small smoke tests or full corpora.
- `build_real_dataset.py` streams JSON/JSONL (no 3M-row memory blowups) and records a `matches` flag per entity for debugging.
- `ner/merge_datasets.py` always re-shuffles merged rows before splitting; use `--train-ratio` to bias train size or `--no-shuffle` for deterministic ordering.
- `ner/ner_train.py` looks for `ner/configs/train_default.json` by default. Duplicate it to create multiple experiment configs (different LR, seeds, checkpoints) without juggling CLI flags. You can set `optim` in the JSON to any Hugging Face `TrainingArguments.optim` value; the script auto-falls back to `adamw_hf` when `torch-xla` is present to avoid fused-optimizer crashes on TPUs.

## Testing / Debugging
- Use `ner/debug.py` (or notebooks) to poke at tokenizer alignment, CLS/SEP handling, etc.
- Parser-related tests live in `tests/`; run `pytest` after touching `inexus/` logic.

## License
MIT unless noted otherwise in subdirectories. Refer to upstream datasets’ licenses if you redistribute derived data.
