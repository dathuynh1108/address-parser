# 0) System deps (if missing Python/pip/venv)
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip git

# 1) Clone your repo (skip if already on the machine)
git clone https://github.com/<your-account>/address-parser.git
cd address-parser

# 2) Create venv + activate
python3 -m venv .venv
source .venv/bin/activate

# 3) Install deps (Torch first so it picks the right CUDA build; omit --extra-index-url if CPU-only)
pip install --upgrade pip
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128   # adjust to your CUDA; remove for CPU
pip install --upgrade -r requirements.txt

# 4) Hugging Face auth (replace with your token)
export HF_TOKEN="hf_xxx"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential --overwrite

# 5) Download dataset into ner/datasets
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="dathuynh1108/ner-address-standard-dataset",
    repo_type="dataset",
    local_dir="ner/datasets",
    local_dir_use_symlinks=False,
    resume_download=True,
)
PY

# 6) Kick off training (uses ner/configs/train_default.json)
python ner/ner_train.py --config ner/configs/train_default.json
