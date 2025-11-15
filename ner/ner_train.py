#!/usr/bin/env python
"""Fine-tune NlpHUST Electra for Vietnamese address NER."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from datasets import DatasetDict, load_dataset
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

DEFAULT_CONFIG = {
  "train_file": "ner/datasets/combined/train.jsonl",
  "eval_file": "ner/datasets/combined/test.jsonl",
  "model_name": "NlpHUST/electra-base-vn",
  "output_dir": "ner/artifacts",
  "epochs": 5,
  "batch_size": 32,
  "learning_rate": 2e-5,
  "weight_decay": 0.01,
  "warmup_ratio": 0.02,
  "gradient_accumulation_steps": 2,
  "max_length": 256,
  "seed": 42,
  "push_to_hub": False,
  "report": True,
  "optim": None,
  "report_to": [],
  "hub_model_id": None,
  "hub_private_repo": False,
}

CONFIG_PATH_KEYS = ("train_file", "eval_file", "output_dir")


def is_xla_available() -> bool:
    try:
        return importlib.util.find_spec("torch_xla") is not None
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ner/configs/train_default.json"),
        help="Path to a JSON config file containing all training hyperparameters.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, object]:
    config = dict(DEFAULT_CONFIG)
    if config_path and config_path.exists():
        user_config = json.loads(config_path.read_text(encoding="utf-8"))
        config.update(user_config)
    for key in CONFIG_PATH_KEYS:
        value = config.get(key)
        if value is not None:
            config[key] = Path(value)
    return config


def ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required dataset file: {path}")


def load_ner_dataset(train_file: Path, eval_file: Optional[Path]) -> DatasetDict:
    data_files: Dict[str, str] = {"train": str(train_file)}
    if eval_file and eval_file.exists():
        data_files["validation"] = str(eval_file)
    dataset = load_dataset("json", data_files=data_files)
    if "validation" not in dataset:
        dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
        dataset = DatasetDict(train=dataset["train"], validation=dataset["test"])
    return dataset


def extract_labels(dataset: DatasetDict) -> List[str]:
    label_set = set()
    for split in dataset.values():
        for tags in split["ner_tags"]:
            label_set.update(tags)
    return sorted(label_set, key=lambda label: (label != "O", label))


def tokenize_and_align(
    dataset: DatasetDict,
    tokenizer: AutoTokenizer,
    label2id: Dict[str, int],
    max_length: int,
) -> DatasetDict:
    sample_split = next(iter(dataset.values()))
    columns_to_remove = [col for col in ("tokens", "ner_tags", "id", "text", "source") if col in sample_split.column_names]

    def _tokenize(batch: Dict[str, List[List[str]]]):
        tokenized = tokenizer(
            batch["tokens"],
            is_split_into_words=True,
            truncation=True,
            padding=False,
            max_length=max_length,
        )
        labels = []
        for idx, label_sequence in enumerate(batch["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=idx)
            previous_word_id = None
            label_ids: List[int] = []
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id != previous_word_id:
                    label_ids.append(label2id[label_sequence[word_id]])
                else:
                    label = label_sequence[word_id]
                    if label.startswith("B-"):
                        label = "I-" + label[2:]
                    label_ids.append(label2id[label])
                previous_word_id = word_id
            labels.append(label_ids)
        tokenized["labels"] = labels
        return tokenized

    return dataset.map(_tokenize, batched=True, remove_columns=columns_to_remove)


def compute_metrics_builder(id2label: Dict[int, str]):
    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        predictions = np.argmax(predictions, axis=-1)
        true_predictions: List[List[str]] = []
        true_labels: List[List[str]] = []
        for prediction, label in zip(predictions, labels):
            pred_tags: List[str] = []
            label_tags: List[str] = []
            for pred_id, label_id in zip(prediction, label):
                if label_id == -100:
                    continue
                pred_tags.append(id2label[pred_id])
                label_tags.append(id2label[label_id])
            if pred_tags:
                true_predictions.append(pred_tags)
                true_labels.append(label_tags)
        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }

    return compute_metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    train_file: Path = config["train_file"]
    eval_file: Optional[Path] = config.get("eval_file")

    ensure_file(train_file)
    dataset = load_ner_dataset(train_file, eval_file)

    label_list = extract_labels(dataset)
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
    tokenized_dataset = tokenize_and_align(dataset, tokenizer, label2id, config["max_length"])

    model = AutoModelForTokenClassification.from_pretrained(
        config["model_name"],
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    optim_name = config.get("optim")
    if optim_name is None and is_xla_available():
        optim_name = "adamw_torch_xla"

    training_kwargs = dict(
        output_dir=str(config["output_dir"]),
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=config["seed"],
        push_to_hub=config["push_to_hub"],
        report_to=config.get("report_to"),
    )
    if optim_name:
        training_kwargs["optim"] = optim_name
    hub_model_id = config.get("hub_model_id")
    if hub_model_id:
        training_kwargs["hub_model_id"] = hub_model_id
    if config.get("hub_private_repo"):
        training_kwargs["hub_private_repo"] = True

    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_builder(id2label),
    )

    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model()
    trainer.state.save_to_json(str(config["output_dir"] / "trainer_state.json"))
    if config["push_to_hub"]:
        commit_message = "Train model"
        eval_f1 = metrics.get("eval_f1")
        if eval_f1 is not None:
            commit_message = f"Train model - eval_f1={eval_f1:.4f}"
        trainer.push_to_hub(commit_message=commit_message)

    print(json.dumps(metrics, indent=2))
    if config["report"]:
        predictions = trainer.predict(tokenized_dataset["validation"])
        predictions_tags = np.argmax(predictions.predictions, axis=-1)
        true_predictions: List[List[str]] = []
        true_labels: List[List[str]] = []
        for prediction, label in zip(predictions_tags, predictions.label_ids):
            pred_tags: List[str] = []
            label_tags: List[str] = []
            for pred_id, label_id in zip(prediction, label):
                if label_id == -100:
                    continue
                pred_tags.append(id2label[pred_id])
                label_tags.append(id2label[label_id])
            if pred_tags:
                true_predictions.append(pred_tags)
                true_labels.append(label_tags)
        print(classification_report(true_labels, true_predictions))


if __name__ == "__main__":
    main()
