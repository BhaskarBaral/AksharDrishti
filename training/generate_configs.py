"""
Generate one PaddleX training config YAML per script, derived from the
official PP-OCRv5 mobile rec template closest to that script (see
prepare_datasets.py's module docstring for the per-script checkpoint/dict
strategy), pointed at the real-data-only dataset this repo just built.

Run after prepare_datasets.py:
  .venv\\Scripts\\python.exe training/generate_configs.py [--epochs N]
"""
import argparse
from pathlib import Path

import yaml

from prepare_datasets import DATASETS_OUT, PRETRAIN_BASE_URL, PRETRAIN_CHECKPOINT, SCRIPTS, TRAINING_ROOT

PADDLEX_CONFIG_DIR = (
    TRAINING_ROOT.parent / ".venv" / "Lib" / "site-packages" / "paddlex"
    / "configs" / "modules" / "text_recognition"
)
CONFIGS_OUT = TRAINING_ROOT / "configs"
OUTPUT_ROOT = TRAINING_ROOT / "output"

# script -> template yaml to derive from (same registered PP-OCRv5 mobile rec
# architecture across all of these; only dataset/pretrain/output differ).
BASE_TEMPLATE = {
    "devanagari": "devanagari_PP-OCRv5_mobile_rec.yaml",
    "tamil": "ta_PP-OCRv5_mobile_rec.yaml",
    "telugu": "te_PP-OCRv5_mobile_rec.yaml",
    "urdu": "arabic_PP-OCRv5_mobile_rec.yaml",
    # No official checkpoint/architecture registration exists for these --
    # reuse the Devanagari template; dataset_dir's own dict.txt (built from
    # this script's real labels) determines the actual head size at train
    # time regardless of which registered model name is used as the base.
    "bengali": "devanagari_PP-OCRv5_mobile_rec.yaml",
    "gujarati": "devanagari_PP-OCRv5_mobile_rec.yaml",
    "kannada": "devanagari_PP-OCRv5_mobile_rec.yaml",
    "malayalam": "devanagari_PP-OCRv5_mobile_rec.yaml",
    "odia": "devanagari_PP-OCRv5_mobile_rec.yaml",
    "punjabi": "devanagari_PP-OCRv5_mobile_rec.yaml",
}


def generate_config(script: str, epochs: int, batch_size: int) -> Path:
    template_path = PADDLEX_CONFIG_DIR / BASE_TEMPLATE[script]
    with open(template_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_dir = (DATASETS_OUT / script).resolve()
    output_dir = (OUTPUT_ROOT / script).resolve()
    pretrain_url = f"{PRETRAIN_BASE_URL}/{PRETRAIN_CHECKPOINT[script]}"
    best_ckpt = f"{output_dir.as_posix()}/best_accuracy/best_accuracy.pdparams"

    cfg["Global"]["mode"] = "train"
    cfg["Global"]["dataset_dir"] = dataset_dir.as_posix()
    # paddlepaddle-gpu (cu126) runs cleanly on this machine's RTX 3050 Ti as
    # long as torch isn't loaded into the same process -- torch's bundled
    # cudnn DLLs collide with paddle's own on Windows (WinError 127,
    # whichever loads second fails). paddlex.engine unconditionally imports
    # torch transitively (via modelscope), so training runs under a separate
    # venv (.venv-train, CPU-only torch) instead of the app's .venv -- see
    # run_all.py's module docstring.
    cfg["Global"]["device"] = "gpu:0"
    cfg["Global"]["output"] = output_dir.as_posix()

    cfg["Train"]["epochs_iters"] = epochs
    cfg["Train"]["batch_size"] = batch_size
    cfg["Train"]["pretrain_weight_path"] = pretrain_url
    cfg["Train"]["eval_interval"] = 1
    cfg["Train"]["save_interval"] = 5

    cfg["Evaluate"]["weight_path"] = best_ckpt
    cfg["Export"]["weight_path"] = best_ckpt
    cfg.pop("Predict", None)

    CONFIGS_OUT.mkdir(parents=True, exist_ok=True)
    out_path = CONFIGS_OUT / f"{script}.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--scripts", nargs="*", default=SCRIPTS)
    args = parser.parse_args()

    for script in args.scripts:
        path = generate_config(script, args.epochs, args.batch_size)
        print(f"{script}: wrote {path}")


if __name__ == "__main__":
    main()
