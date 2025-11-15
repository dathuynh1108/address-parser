import os
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="datasets",
    repo_id="dathuynh1108/ner-address-standard-dataset",
    repo_type="dataset",
)
