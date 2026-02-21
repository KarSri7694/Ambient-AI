from huggingface_hub import snapshot_download
from pathlib import Path
import os

current_dir = Path(__file__).parent
project_root = current_dir.parent
os.chdir(project_root)

snapshot_download("Qwen/Qwen2.5-VL-3B-Instruct", local_dir="qwen-2-5-3b-vl-instruct")
