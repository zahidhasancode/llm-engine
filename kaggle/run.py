"""Kaggle entry point: pin libraries, find the code, test it, then run the real benchmark.

The code comes from the private dataset `llm-engine-code`, built by kaggle/push.sh.
Weights download from Hugging Face at a fixed commit. Everything worth keeping is written
to /kaggle/working/outputs.
"""

import glob
import os
import shutil
import subprocess
import sys

# Set before anything imports torch. The T4 machine has two GPUs and this run uses one.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

WORK = "/kaggle/working"
OUTPUTS = f"{WORK}/outputs"

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

PINNED = [
    "torch==2.15.0",
    "transformers==5.22.1",
    "safetensors==0.8.0",
    "pytest==9.2.0",
]


def sh(*command: str, **kwargs) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.check_call(list(command), **kwargs)


def find_code() -> str:
    """Locate the uploaded code: a folder holding engine/model.py, or a zip containing one."""
    hits = glob.glob("/kaggle/input/**/engine/model.py", recursive=True)
    if hits:
        return os.path.dirname(os.path.dirname(hits[0]))
    for archive in glob.glob("/kaggle/input/**/*.zip", recursive=True):
        shutil.unpack_archive(archive, f"{WORK}/unzipped")
        hits = glob.glob(f"{WORK}/unzipped/**/engine/model.py", recursive=True)
        if hits:
            return os.path.dirname(os.path.dirname(hits[0]))
    sh("find", "/kaggle/input", "-maxdepth", "5")
    raise SystemExit("code not found under /kaggle/input")


def main() -> None:
    sh(sys.executable, "-m", "pip", "install", "-q", *PINNED)
    sh("nvidia-smi")

    repo = f"{WORK}/repo"
    shutil.copytree(find_code(), repo, dirs_exist_ok=True)
    os.chdir(repo)
    sh(sys.executable, "-m", "pytest", "-q", "tests")

    from huggingface_hub import snapshot_download

    weights = snapshot_download(MODEL, revision=REVISION)

    sh(
        sys.executable,
        "benchmarks/real_model.py",
        "--model-path", weights,
        "--model-name", MODEL,
        "--revision", REVISION,
        "--output", f"{OUTPUTS}/real_model_t4.json",
        env=os.environ | {"PYTHONPATH": repo},
    )
    print("finished; outputs in", OUTPUTS)


if __name__ == "__main__":
    main()
