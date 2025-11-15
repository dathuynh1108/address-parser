import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

HF_TOKEN_ENV = "HF_TOKEN"
DEFAULT_DATASET_DIR = Path("datasets")
DEFAULT_MODEL_DIR = Path("ner/artifacts")
DEFAULT_DATASET_REPO = "dathuynh1108/ner-address-standard-dataset"
DEFAULT_MODEL_REPO = "dathuynh1108/ner-address-electra-base-vn"


def ensure_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Folder not found: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"Expected directory but found file: {resolved}")
    return resolved


def upload(api: HfApi, folder: Path, repo_id: str, repo_type: str) -> None:
    path = ensure_dir(folder)
    api.upload_folder(folder_path=str(path), repo_id=repo_id, repo_type=repo_type)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload dataset/model folders to Hugging Face Hub.")
    parser.add_argument(
        "--upload",
        choices=("dataset", "model", "both"),
        default="dataset",
        help="Which artifact(s) to upload.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Local dataset folder to upload.",
    )
    parser.add_argument(
        "--dataset-repo",
        default=DEFAULT_DATASET_REPO,
        help="Target dataset repo id on Hugging Face.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Local model/artifact folder to upload.",
    )
    parser.add_argument(
        "--model-repo",
        default=DEFAULT_MODEL_REPO,
        help="Target model repo id on Hugging Face.",
    )
    parser.add_argument(
        "--token-env",
        default=HF_TOKEN_ENV,
        help="Environment variable name storing the Hugging Face token.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.getenv(args.token_env)
    if not token:
        raise EnvironmentError(f"Missing Hugging Face token. Set the {args.token_env} environment variable.")

    api = HfApi(token=token)

    if args.upload in ("dataset", "both"):
        upload(api, args.dataset_dir, args.dataset_repo, "dataset")

    if args.upload in ("model", "both"):
        upload(api, args.model_dir, args.model_repo, "model")


if __name__ == "__main__":
    main()
