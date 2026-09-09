"""
Spot-check a fine-tuned recognition model: predicts on either a single image
or a random sample of that script's val.txt, and prints prediction vs.
ground truth. Runs on CPU by default so it's safe alongside a GPU training
run in progress (.venv-train) -- this uses the app's .venv, which only has
CPU paddlepaddle installed.

Usage:
  .venv\\Scripts\\python.exe training/test_predict.py devanagari
  .venv\\Scripts\\python.exe training/test_predict.py devanagari --n 20
  .venv\\Scripts\\python.exe training/test_predict.py devanagari --image path\\to\\crop.png
"""
import argparse
import random
import re
import sys
from pathlib import Path

from generate_configs import BASE_TEMPLATE
from prepare_datasets import DATASETS_OUT, TRAINING_ROOT

OUTPUT_ROOT = TRAINING_ROOT / "output"

# script -> registered PaddleX model name (must match what the config was
# generated from -- see generate_configs.py's BASE_TEMPLATE).
MODEL_NAME = {
    script: template.removesuffix(".yaml") for script, template in BASE_TEMPLATE.items()
}

# See app/engines/paddleocr_finetuned_engine.py's module docstring: our
# fine-tuned Urdu model emits characters in reversed order (PaddleX's modern
# TextRecognition predictor never applies the RTL un-reversal the older
# training/eval tooling does), so undo it here too for accurate spot-checks.
_RTL_SCRIPTS = {"urdu"}


def _pred_reverse(pred: str) -> str:
    pred_re = []
    c_current = ""
    for c in pred:
        if not re.search("[a-zA-Z0-9 :*./%+-]", c):
            if c_current:
                pred_re.append(c_current)
            pred_re.append(c)
            c_current = ""
        else:
            c_current += c
    if c_current:
        pred_re.append(c_current)
    return "".join(pred_re[::-1])


def load_val_samples(script: str, n: int):
    val_path = DATASETS_OUT / script / "val.txt"
    lines = val_path.read_text(encoding="utf-8").splitlines()
    pairs = [tuple(line.split("\t", 1)) for line in lines if "\t" in line]
    random.seed(0)
    return random.sample(pairs, min(n, len(pairs)))


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("script")
    parser.add_argument("--n", type=int, default=10, help="random val samples to test")
    parser.add_argument("--image", help="test a single image instead of sampling val.txt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model_dir = OUTPUT_ROOT / args.script / "best_accuracy" / "inference"
    if not model_dir.exists():
        raise SystemExit(f"No trained model at {model_dir} -- has {args.script} finished training?")

    from paddleocr import TextRecognition

    predictor = TextRecognition(
        model_name=MODEL_NAME[args.script], model_dir=str(model_dir), device=args.device
    )

    samples = [(args.image, None)] if args.image else load_val_samples(args.script, args.n)

    correct = 0
    for img_path, gt in samples:
        result = list(predictor.predict(img_path))[0]
        pred, score = result["rec_text"], result["rec_score"]
        if args.script in _RTL_SCRIPTS:
            pred = _pred_reverse(pred)
        if gt is None:
            print(f"pred={pred!r}  score={score:.3f}")
            continue
        ok = pred == gt
        correct += ok
        print(f"{'OK ' if ok else 'XX '}pred={pred!r:20}  score={score:.3f}  gt={gt!r}")

    if not args.image:
        print(f"\n{correct}/{len(samples)} exact matches")


if __name__ == "__main__":
    main()
