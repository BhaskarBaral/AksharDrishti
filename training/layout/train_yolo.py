"""
Fine-tune the IndicDLP DocLayout-YOLO detector (Zenodo 10.5281/zenodo.15881917,
CC-BY-4.0, trained by the IndicDLP authors on ~120k real Indic pages) on the
dataset prepare_yolo.py builds: real IndicDLP pages + Stage 2 synthetic pages,
adding seal / stamp / signature to IndicDLP's 42 classes.

The head is re-sized from 42 to 45 classes; every other layer starts from the
IndicDLP weights. Output: training/layout/runs/<name>/weights/best.pt.

Usage:
  .venv\\Scripts\\python.exe training/layout/train_yolo.py --epochs 6 --batch 4
"""
import argparse
from pathlib import Path

import doclayout_yolo.engine.trainer as yolo_trainer
from doclayout_yolo import YOLOv10

# The trainer's AMP self-test downloads a separate yolov8n.pt to compare
# fp16/fp32 outputs; skip it (AMP is known-good on this CUDA 12 / RTX setup).
yolo_trainer.check_amp = lambda model: True

HERE = Path(__file__).resolve().parent
BASE_WEIGHTS = Path(r"C:\Users\Legion\Desktop\AksharDristhi\datasets\external\indicdlp-detector-zenodo"
                    r"\doclayout_yolo_indicdlp.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4)
    # 1024 (the IndicDLP authors' size) overflows a 4 GB GPU into shared
    # memory and runs ~4x slower; 800 fits.
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--name", default="indicdlp_ft")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    model = YOLOv10(str(BASE_WEIGHTS))
    model.train(
        data=str(HERE / "yolo" / "data.yaml"), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=0, workers=args.workers, project=str(HERE / "runs"), name=args.name, exist_ok=True,
        # Fine-tuning a converged detector: low LR, short warmup, no mosaic
        # (documents are single pages -- mosaic tiles them into nonsense).
        optimizer="SGD", lr0=0.004, warmup_epochs=0.5, mosaic=0.0, close_mosaic=0,
        fliplr=0.0, amp=True, plots=False, patience=100,
    )


if __name__ == "__main__":
    main()
