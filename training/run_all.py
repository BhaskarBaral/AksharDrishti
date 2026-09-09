"""
Sequentially fine-tune PaddleOCR recognition for every script with a
generated config in training/configs/. One script at a time on the GPU,
so they don't fight over the same 4GB of VRAM. Skips a script if its best
checkpoint already exists, so a killed/interrupted run can just be
restarted.

Runs under training/../.venv-train, a separate venv from the app's
.venv: paddlex.engine unconditionally imports torch (via modelscope),
and torch's bundled cudnn DLLs collide with paddlepaddle-gpu's own on
Windows (whichever loads second fails with WinError 127) -- see
generate_configs.py's device note. .venv-train carries a CPU-only torch
build (no bundled cudnn DLLs, so nothing to collide) alongside
paddlepaddle-gpu.

Usage:
  .venv-train\\Scripts\\python.exe training/run_all.py [script ...]
"""
import subprocess
import sys
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = TRAINING_ROOT / "configs"
OUTPUT_ROOT = TRAINING_ROOT / "output"
PYTHON = TRAINING_ROOT.parent / ".venv-train" / "Scripts" / "python.exe"

DEFAULT_ORDER = [
    "devanagari", "tamil", "telugu", "urdu",  # official-checkpoint scripts first
    "bengali", "gujarati", "kannada", "malayalam", "odia", "punjabi",
]


def already_done(script: str) -> bool:
    return (OUTPUT_ROOT / script / "best_accuracy" / "best_accuracy.pdparams").exists()


def main():
    scripts = sys.argv[1:] or DEFAULT_ORDER
    for script in scripts:
        config_path = CONFIGS_DIR / f"{script}.yaml"
        if not config_path.exists():
            print(f"[skip] {script}: no config at {config_path}")
            continue
        if already_done(script):
            print(f"[skip] {script}: best_accuracy.pdparams already exists")
            continue

        print(f"\n{'=' * 60}\n[train] {script}\n{'=' * 60}", flush=True)
        result = subprocess.run(
            [str(PYTHON), str(TRAINING_ROOT / "train_one.py"), "-c", str(config_path)],
        )
        status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
        print(f"[{status}] {script}", flush=True)


if __name__ == "__main__":
    main()
