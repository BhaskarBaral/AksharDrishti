"""
Train a page-layout detector (header / form-field / table / table-cell /
seal / stamp / signature ...) on the synthetic pages' layout sidecars.

Input is the COCO export from the dataset repo's scripts/layout_to_coco.py:
  python scripts/layout_to_coco.py synthetic-layout --out <data>/coco_extras.json
  python scripts/layout_to_coco.py synthetic --only-stamps-labelled --out <data>/coco_synthetic.json
(--only-stamps-labelled: moderate/severe pages in synthetic/ may carry an
unlabelled seal, which would teach the model that seals are background.)

Model: torchvision Faster R-CNN, MobileNetV3-Large FPN, COCO-pretrained --
already in the app's .venv (no new dependency) and small enough for a 4 GB GPU.
The RPN anchors get extra flat aspect ratios (h/w down to 0.1) because form
rows and table headers are long thin boxes the default 0.5/1/2 anchors miss.

Splits reuse layout_to_coco.py's own filename-hash buckets (80/10/10), so the
same page is always in the same split across both corpora.

Usage:
  .venv\\Scripts\\python.exe training/layout/train_layout.py --epochs 6
  .venv\\Scripts\\python.exe training/layout/train_layout.py --eval-only
"""
import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (FasterRCNN_MobileNet_V3_Large_FPN_Weights,
                                          fasterrcnn_mobilenet_v3_large_fpn)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import RPNHead
from torchvision.transforms.functional import to_tensor

HERE = Path(__file__).resolve().parent
DATASET_ROOT = Path(r"C:\Users\Legion\Desktop\Akshar_dristhii_code")
SOURCES = [  # (coco file, image root)
    (HERE / "data" / "coco_extras.json", DATASET_ROOT / "synthetic-layout"),
    (HERE / "data" / "coco_synthetic.json", DATASET_ROOT / "synthetic"),
]
OUT_DIR = HERE / "output"

CLASSES = ["header", "form-field", "field-label", "field-value", "table", "table-header",
           "table-cell", "seal", "stamp", "signature"]
SCALE = 0.5  # pages are 1600x2200; 800x1100 keeps table cells >= ~20 px tall
ANCHOR_SIZES = ((32, 64, 128, 256, 512),) * 3
ANCHOR_RATIOS = ((0.1, 0.25, 0.5, 1.0, 2.0),) * 3


def bucket(name: str) -> str:
    # Same as layout_to_coco.py's bucket().
    v = int(hashlib.md5(name.encode()).hexdigest(), 16) % 10
    return "train" if v < 8 else ("val" if v == 8 else "test")


def load_pages() -> dict[str, list[dict]]:
    """-> {split: [{path, doctype, script, quality, boxes, labels}]}"""
    by_split = defaultdict(list)
    for coco_path, root in SOURCES:
        coco = json.loads(coco_path.read_text(encoding="utf-8"))
        cat = {c["id"]: CLASSES.index(c["name"]) + 1 for c in coco["categories"]}
        anns = defaultdict(list)
        for a in coco["annotations"]:
            anns[a["image_id"]].append(a)
        for im in coco["images"]:
            boxes = [[x, y, x + w, y + h] for x, y, w, h in (a["bbox"] for a in anns[im["id"]])]
            keep = [i for i, b in enumerate(boxes) if b[2] - b[0] > 2 and b[3] - b[1] > 2]
            by_split[bucket(Path(im["file_name"]).name)].append({
                "path": str(root / im["file_name"]), "doctype": im["doctype"],
                "script": im["script"], "quality": im["quality"], "source": root.name,
                "boxes": [boxes[i] for i in keep],
                "labels": [cat[anns[im["id"]][i]["category_id"]] for i in keep],
            })
    return by_split


def subsample(pages: list[dict], per_source_doctype: int, seed: int) -> list[dict]:
    """Cap pages per (source, doctype) so the 47k-page corpus doesn't drown the
    2.7k pages that carry labelled seals/stamps/signatures."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for p in pages:
        groups[(p["source"], p["doctype"])].append(p)
    out = []
    for g in groups.values():
        rng.shuffle(g)
        out += g[:per_source_doctype]
    rng.shuffle(out)
    return out


class LayoutPages(Dataset):
    def __init__(self, pages):
        self.pages = pages

    def __len__(self):
        return len(self.pages)

    def __getitem__(self, i):
        p = self.pages[i]
        im = Image.open(p["path"]).convert("RGB")
        im = im.resize((round(im.width * SCALE), round(im.height * SCALE)), Image.BILINEAR)
        boxes = torch.tensor(p["boxes"], dtype=torch.float32).reshape(-1, 4) * SCALE
        return to_tensor(im), {"boxes": boxes, "labels": torch.tensor(p["labels"], dtype=torch.int64)}


def collate(batch):
    return tuple(zip(*batch))


def build_model(pretrained: bool = True):
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1 if pretrained else None,
        weights_backbone=None, min_size=800, max_size=1100, box_detections_per_img=300)
    # Images arrive pre-scaled by SCALE, so this makes the internal resize a
    # no-op for 800x1100 pages.
    model.rpn.anchor_generator = AnchorGenerator(ANCHOR_SIZES, ANCHOR_RATIOS)
    model.rpn.head = RPNHead(model.backbone.out_channels,
                             model.rpn.anchor_generator.num_anchors_per_location()[0])
    model.roi_heads.box_predictor = FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features, len(CLASSES) + 1)
    return model


def box_iou(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def average_precision(preds, gts, iou_thr=0.5):
    """VOC-style all-point AP@iou_thr per class. preds/gts: per-image
    {"boxes", "labels"(, "scores")} numpy dicts."""
    out = {}
    for c in range(1, len(CLASSES) + 1):
        dets, n_gt = [], 0
        for img_i, (p, g) in enumerate(zip(preds, gts)):
            n_gt += int((g["labels"] == c).sum())
            m = p["labels"] == c
            dets += [(s, img_i, b) for s, b in zip(p["scores"][m], p["boxes"][m])]
        if n_gt == 0:
            continue
        dets.sort(key=lambda d: -d[0])
        used = defaultdict(set)
        tp = np.zeros(len(dets))
        for k, (_, img_i, b) in enumerate(dets):
            g = gts[img_i]
            gb = g["boxes"][g["labels"] == c]
            if len(gb):
                ious = box_iou(b[None], gb)[0]
                j = int(ious.argmax())
                if ious[j] >= iou_thr and j not in used[img_i]:
                    tp[k] = 1
                    used[img_i].add(j)
        ctp = np.cumsum(tp)
        rec = ctp / n_gt
        prec = ctp / np.arange(1, len(dets) + 1) if len(dets) else np.zeros(0)
        mrec = np.concatenate([[0], rec, [1]])
        mpre = np.concatenate([[0], prec, [0]])
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        out[CLASSES[c - 1]] = (float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum()), n_gt)
    return out


@torch.no_grad()
def evaluate(model, pages, device, batch_size=4):
    model.eval()
    loader = DataLoader(LayoutPages(pages), batch_size=batch_size, num_workers=4, collate_fn=collate)
    preds, gts = [], []
    for ims, tgts in loader:
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            outs = model([im.to(device) for im in ims])
        for o, t in zip(outs, tgts):
            preds.append({k: o[k].float().cpu().numpy() for k in ("boxes", "labels", "scores")})
            gts.append({k: t[k].numpy() for k in ("boxes", "labels")})
    return average_precision(preds, gts)


def report(ap: dict, title: str) -> float:
    print(f"\n{title}")
    for c, (v, n) in ap.items():
        print(f"  {c:13s} AP@0.5={v:.3f}  (n={n})")
    m = float(np.mean([v for v, _ in ap.values()])) if ap else 0.0
    print(f"  {'mAP@0.5':13s} {m:.3f}")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--train-per-group", type=int, default=400, help="pages per (source, doctype)")
    ap.add_argument("--eval-per-group", type=int, default=40)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = OUT_DIR / "layout_frcnn_mnv3.pt"

    splits = load_pages()
    train = subsample(splits["train"], args.train_per_group, args.seed)
    val = subsample(splits["val"], args.eval_per_group, args.seed)
    test = subsample(splits["test"], args.eval_per_group, args.seed)
    print(f"pages: train={len(train)} val={len(val)} test={len(test)}")
    print("train labels:", dict(Counter(CLASSES[l - 1] for p in train for l in p["labels"])))

    model = build_model().to(device)
    if not args.eval_only:
        loader = DataLoader(LayoutPages(train), batch_size=args.batch_size, shuffle=True,
                            num_workers=4, collate_fn=collate, persistent_workers=True)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
        total = args.epochs * len(loader)
        warmup = min(300, len(loader))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + np.cos(np.pi * s / total)))
        scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
        best = -1.0
        for epoch in range(args.epochs):
            model.train()
            t0, running = time.time(), 0.0
            for step, (ims, tgts) in enumerate(loader):
                ims = [im.to(device) for im in ims]
                tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
                with torch.autocast("cuda", enabled=device.type == "cuda"):
                    loss = sum(model(ims, tgts).values())
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 10.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                running += loss.item()
                if step % 100 == 0:
                    print(f"epoch {epoch + 1} step {step}/{len(loader)} loss {running / (step + 1):.3f} "
                          f"({time.time() - t0:.0f}s)", flush=True)
            m = report(evaluate(model, val, device), f"epoch {epoch + 1} val")
            if m > best:
                best = m
                torch.save({"model": model.state_dict(), "classes": CLASSES, "scale": SCALE,
                            "anchor_ratios": ANCHOR_RATIOS, "epoch": epoch + 1, "val_map50": m}, ckpt)
                print(f"  saved {ckpt.name} (val mAP@0.5 {m:.3f})", flush=True)

    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    test_ap = evaluate(model, test, device)
    m = report(test_ap, "held-out test (synthetic)")
    by_source = {}
    for src in ("synthetic-layout", "synthetic"):
        sub = [p for p in test if p["source"] == src]
        by_source[src] = report(evaluate(model, sub, device), f"test, {src} only ({len(sub)} pages)")
    (OUT_DIR / "metrics.json").write_text(json.dumps({
        "test_map50": m, "test_ap50": {c: v for c, (v, _) in test_ap.items()},
        "test_pages": len(test), "by_source_map50": by_source}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
