"""Kaggle entry point for step 10. push.sh sets STAGE to "smoke", "benchmark" or "gap".

smoke:      short checks on the GPU, including whether vLLM runs here at all.
benchmark:  the full run. Only start it after the smoke test has passed.
gap:        splits the gap to vLLM into measured pieces. See benchmarks/speed_gap.py.

vLLM pins an older torch and transformers than this repository, so it gets its own virtual
environment. Each system runs as its own process. Outputs go to /kaggle/working/outputs.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time

# Set before anything imports torch. The T4 machine has two GPUs and this run uses one.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

STAGE = "smoke"

WORK = "/kaggle/working"
OUTPUTS = f"{WORK}/outputs"
REPO = "/tmp/repo"
VLLM_ENV = "/tmp/vllm-env"
VLLM_BIN = f"{VLLM_ENV}/bin/vllm"

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

PINNED = [
    "torch==2.15.0",
    "transformers==5.22.1",
    "safetensors==0.8.0",
    "pytest==9.2.0",
    "fastapi==0.162.0",
    "uvicorn==0.54.0",
    "httpx==0.28.1",
]
VLLM_PINNED = ["vllm==0.32.0"]


def sh(*command: str, check: bool = True, **kwargs) -> int:
    print("+", " ".join(command), flush=True)
    result = subprocess.run(list(command), **kwargs)
    if check and result.returncode != 0:
        raise SystemExit(f"failed with {result.returncode}: {' '.join(command)}")
    return result.returncode


def find_code() -> str:
    hits = glob.glob("/kaggle/input/**/engine/model.py", recursive=True)
    if hits:
        return os.path.dirname(os.path.dirname(hits[0]))
    for archive in glob.glob("/kaggle/input/**/*.zip", recursive=True):
        shutil.unpack_archive(archive, "/tmp/unzipped")
        hits = glob.glob("/tmp/unzipped/**/engine/model.py", recursive=True)
        if hits:
            return os.path.dirname(os.path.dirname(hits[0]))
    sh("find", "/kaggle/input", "-maxdepth", "5")
    raise SystemExit("code not found under /kaggle/input")


def install() -> None:
    sh(sys.executable, "-m", "pip", "install", "-q", *PINNED)
    # The image's torchvision is built for its own torch, and transformers imports it.
    sh(sys.executable, "-m", "pip", "uninstall", "-q", "-y", "torchvision", "torchaudio")
    sh("nvidia-smi")
    sh(sys.executable, "-c", "import torch; print(torch.__version__, torch.version.cuda); "
       "assert torch.cuda.device_count() == 1; print(torch.cuda.get_device_name(0))")


def install_vllm() -> bool:
    """Returns False instead of stopping, so the smoke test can report the failure."""
    # The image's Python has no ensurepip, so `python -m venv` cannot work here.
    sh(sys.executable, "-m", "pip", "install", "-q", "virtualenv==20.31.2")
    if sh(sys.executable, "-m", "virtualenv", VLLM_ENV, check=False) != 0:
        return False
    pip = f"{VLLM_ENV}/bin/pip"
    if sh(pip, "install", "-q", *VLLM_PINNED, check=False) != 0:
        return False
    return sh(f"{VLLM_ENV}/bin/python", "-c",
              "import torch, vllm; print(vllm.__version__, torch.__version__, torch.version.cuda); "
              "assert torch.cuda.is_available()", check=False) == 0


def versions() -> dict:
    def freeze(python: str) -> dict:
        lines = subprocess.run([python, "-m", "pip", "freeze"], capture_output=True, text=True)
        wanted = ("torch==", "transformers==", "vllm==", "fastapi==", "uvicorn==", "httpx==")
        return [line for line in lines.stdout.splitlines() if line.startswith(wanted)]

    info = {"engine_env": freeze(sys.executable)}
    if os.path.exists(f"{VLLM_ENV}/bin/python"):
        info["vllm_env"] = freeze(f"{VLLM_ENV}/bin/python")
    info["gpu"] = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader",
         "-i", "0"], capture_output=True, text=True).stdout.strip()
    info["cpu"] = subprocess.run("lscpu | grep -E 'Model name|^CPU\\(s\\)'", shell=True,
                                 capture_output=True, text=True).stdout
    return info


def weights() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL, revision=REVISION)


def smoke(path: str, vllm_ok: bool) -> None:
    sys.path.insert(0, REPO)
    import httpx

    from benchmarks.serving_benchmark import GREEDY, Server, systems
    from scripts import loadtest

    class Args:
        model_path = path
        vllm_bin = VLLM_BIN
        dtype = "float16"
        max_length = 2048
        slots = 16
        prefill_tokens_per_step = 256

    report = {"vllm_installed": vllm_ok, "systems": {}, "versions": versions()}
    question = [{"role": "user", "content": "Why is the sky blue? Answer in one sentence."}]
    for system in systems(Args):
        if system.name == "vllm" and not vllm_ok:
            continue
        entry = {}
        server = Server(system, f"{OUTPUTS}/{system.name}_smoke.log")
        try:
            started = time.monotonic()
            server.wait_until_ready(timeout=600)
            entry["start_seconds"] = time.monotonic() - started
            body = {"model": "qwen", "messages": question, "max_tokens": 40, "temperature": 0,
                    **json.loads(GREEDY)}
            reply = httpx.post(f"{server.url}/v1/chat/completions", json=body, timeout=120)
            entry["status"] = reply.status_code
            entry["reply"] = reply.text[:600]
            entry["answer"] = reply.json()["choices"][0]["message"]["content"]
            result = loadtest.main([
                "--base-url", server.url, "--model", "qwen", "--prompts", "data/prompts.jsonl",
                "--mode", "closed", "--users", "4", "--requests", "12", "--max-tokens", "32",
                "--ignore-eos", "--extra-body", GREEDY,
            ])
            entry["load"] = result["summary"]
        except Exception as error:
            entry["error"] = f"{type(error).__name__}: {error}"
        finally:
            server.stop()
        report["systems"][system.name] = entry
        with open(f"{OUTPUTS}/smoke.json", "w") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)

    env = os.environ | {"PYTHONPATH": REPO}
    report["step_time_exit"] = sh(
        sys.executable, "benchmarks/step_time.py", "--model-path", path,
        "--output", f"{OUTPUTS}/step_time_smoke.json",
        "--batches", "1", "8", "--steps", "16", "--repeats", "1", env=env, check=False)
    if vllm_ok:
        report["vllm_step_time_exit"] = sh(
            f"{VLLM_ENV}/bin/python", "benchmarks/vllm_step_time.py", "--model-path", path,
            "--output", f"{OUTPUTS}/vllm_step_time_smoke.json",
            "--batches", "1", "8", "--steps", "16", "--repeats", "1", check=False)

    with open(f"{OUTPUTS}/smoke.json", "w") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    print("smoke finished", flush=True)


def benchmark(path: str) -> None:
    env = os.environ | {"PYTHONPATH": REPO}
    with open(f"{OUTPUTS}/versions.json", "w") as file:
        json.dump(versions(), file, indent=2)
    sh(sys.executable, "benchmarks/step_time.py", "--model-path", path,
       "--output", f"{OUTPUTS}/step_time.json", env=env, check=False)
    sh(f"{VLLM_ENV}/bin/python", "benchmarks/vllm_step_time.py", "--model-path", path,
       "--output", f"{OUTPUTS}/vllm_step_time.json", check=False)
    sh(sys.executable, "benchmarks/serving_benchmark.py", "--model-path", path,
       "--vllm-bin", VLLM_BIN, "--output-dir", OUTPUTS, env=env)
    print("benchmark finished", flush=True)


def gap(path: str) -> None:
    env = os.environ | {"PYTHONPATH": REPO}
    with open(f"{OUTPUTS}/versions.json", "w") as file:
        json.dump(versions(), file, indent=2)
    sh(sys.executable, "benchmarks/speed_gap.py", "--model-path", path, "--output-dir", OUTPUTS,
       env=env)
    variants = {
        "default": [],
        "eager": ["--enforce-eager"],
        "flex": ["--attention-backend", "FLEX_ATTENTION"],
        "flex_eager": ["--enforce-eager", "--attention-backend", "FLEX_ATTENTION"],
    }
    for name, flags in variants.items():
        sh(f"{VLLM_ENV}/bin/python", "benchmarks/vllm_step_time.py", "--model-path", path,
           "--output", f"{OUTPUTS}/vllm_step_time_{name}.json",
           "--batches", "1", "16", "32", "64", "128", *flags, check=False)
    print("gap finished", flush=True)


def main() -> None:
    os.makedirs(OUTPUTS, exist_ok=True)
    install()
    vllm_ok = install_vllm()
    print("vllm installed:", vllm_ok, flush=True)
    shutil.copytree(find_code(), REPO, dirs_exist_ok=True)
    os.chdir(REPO)
    sh(sys.executable, "-m", "pytest", "-q", "tests")
    path = weights()
    if STAGE == "smoke":
        smoke(path, vllm_ok)
    else:
        if not vllm_ok:
            raise SystemExit("vLLM did not install")
        gap(path) if STAGE == "gap" else benchmark(path)


if __name__ == "__main__":
    main()
