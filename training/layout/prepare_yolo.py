"""
Build the YOLO-format dataset for fine-tuning the IndicDLP DocLayout-YOLO
detector (train_yolo.py) on real + synthetic pages.

Classes: IndicDLP's own 42 (ids 0-41, from the IndicDLP repo's indic.yaml --
same order as the Zenodo checkpoint's model.names) plus three of ours that
IndicDLP doesn't have: 42 seal, 43 stamp, 44 signature.

Sources:
  real train  IndicDLP train shards cached by huggingface_hub (MIT; gated
              download, already on disk). Ground-truth boxes as shipped.
              Pages that also appear in the dataset repo's real
              complex-testset/ (a test set -- never train on it) are skipped
              by pixel hash.
  real val    IndicDLP Odia validation shard (real held-out split).
  real test   IndicDLP Odia test shard -- only for evaluate_yolo.py.
  synthetic   Stage 2 synthetic-layout/ pages (layout_to_coco.py's own
              80/10/10 filename-hash split). Their own exact region labels,
              mapped onto IndicDLP's classes (SYNTHETIC_TO_INDICDLP) so form
              rows and tables are positives, not background. (Pseudo-labels
              from the base IndicDLP model were tried first and dropped: it
              finds almost nothing on these pages -- no rows, no tables -- so
              they'd have taught the fine-tune to ignore both.) A third of the synthetic
              train pages also get 1-3 real IndicDLP `figure` crops pasted on
              (labelled figure) -- the hard negatives the synthetic-only model
              lacked, which is why it called drawings signatures/seals.

Usage:
  .venv\\Scripts\\python.exe training/layout/prepare_yolo.py
"""
import hashlib
import io
import json
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE / "yolo"
HF_INDICDLP = Path.home() / ".cache/huggingface/hub/datasets--ai4bharat--indicdlp/snapshots"
ODIA_SHARD = Path(r"C:\Users\Legion\Desktop\AksharDristhi\datasets\external\indicdlp-odia-shard14")
BASE_WEIGHTS = Path(r"C:\Users\Legion\Desktop\AksharDristhi\datasets\external\indicdlp-detector-zenodo"
                    r"\doclayout_yolo_indicdlp.pt")
DATASET_ROOT = Path(r"C:\Users\Legion\Desktop\Akshar_dristhii_code")
COMPLEX_TESTSET = DATASET_ROOT / "complex-testset" / "pages"
SYNTHETIC_LAYOUT = DATASET_ROOT / "synthetic-layout"

INDICDLP_NAMES = [
    "advertisement", "answer", "author", "chapter-title", "contact-info", "dateline", "figure",
    "figure-caption", "first-level-question", "flag", "folio", "footer", "footnote", "formula", "header",
    "headline", "index", "jumpline", "options", "ordered-list", "page-number", "paragraph",
    "placeholder-text", "quote", "reference", "second-level-question", "section-title", "sidebar",
    "sub-headline", "sub-ordered-list", "sub-section-title", "subsub-ordered-list", "subsub-section-title",
    "sub-unordered-list", "subsub-headline", "subsub-unordered-list", "table", "table-caption",
    "table-of-contents", "third-level-question", "unordered-list", "website-link",
]
EXTRA_NAMES = ["seal", "stamp", "signature"]
NAMES = INDICDLP_NAMES + EXTRA_NAMES
FIGURE = INDICDLP_NAMES.index("figure")
EXTRA_ID = {n: len(INDICDLP_NAMES) + i for i, n in enumerate(EXTRA_NAMES)}
MAX_SIDE = 1280
# Synthetic sidecar class -> IndicDLP class. placeholder-text is IndicDLP's
# Forms-domain field class; table-header/table-cell have no IndicDLP
# equivalent (it labels the table as one region) so they're dropped.
SYNTHETIC_TO_INDICDLP = {
    "header": "section-title", "form-field": "placeholder-text", "field-value": "placeholder-text",
    "field-label": "paragraph", "table": "table",
}


def pixel_hash(im: Image.Image) -> str:
    return hashlib.md5(im.convert("RGB").tobytes()).hexdigest()


def save(im: Image.Image, split: str, stem: str, boxes: list[tuple[int, float, float, float, float]]):
    """boxes: (cls, x0, y0, x1, y1) in im's pixel coords."""
    w, h = im.size
    scale = min(1.0, MAX_SIDE / max(w, h))
    if scale < 1:
        im = im.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    im.convert("RGB").save(OUT / "images" / split / f"{stem}.jpg", quality=90)
    lines = []
    for c, x0, y0, x1, y1 in boxes:
        x0, x1 = max(0.0, min(x0, x1)), min(float(w), max(x0, x1))
        y0, y1 = max(0.0, min(y0, y1)), min(float(h), max(y0, y1))
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        lines.append(f"{c} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
    (OUT / "labels" / split / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")


def iter_parquet(path: Path):
    # Small batches: a whole row group of page images doesn't fit next to the
    # demo's OCR models in RAM.
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
        for row in batch.to_pylist():
            im = Image.open(io.BytesIO(row["image"]["bytes"]))
            im.load()
            boxes = [(c, x, y, x + bw, y + bh) for (x, y, bw, bh), c in zip(row["bboxes"], row["category_ids"])]
            yield im, boxes


def bucket(name: str) -> str:
    # Same as the dataset repo's layout_to_coco.py.
    v = int(hashlib.md5(name.encode()).hexdigest(), 16) % 10
    return "train" if v < 8 else ("val" if v == 8 else "test")


def main():
    rng = random.Random(42)
    for split in ("train", "val", "test"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    excluded, excluded_sizes = set(), set()
    for p in COMPLEX_TESTSET.glob("*.png"):
        im = Image.open(p)
        excluded.add(pixel_hash(im))
        excluded_sizes.add(im.size)
    print(f"complex-testset pages to exclude: {len(excluded)}", flush=True)

    # --- real train (+ collect figure crops for hard negatives) ---
    figure_crops: list[Image.Image] = []
    n_train = n_skipped = 0
    for shard in sorted(HF_INDICDLP.glob("*/data/train-*.parquet")):
        for im, boxes in iter_parquet(shard):
            # Hash only same-size pages -- hashing every page costs too much RAM.
            if im.size in excluded_sizes and pixel_hash(im) in excluded:
                n_skipped += 1
                continue
            save(im, "train", f"real_{n_train:05d}", boxes)
            n_train += 1
            for c, x0, y0, x1, y1 in boxes:
                # A few IndicDLP boxes run far outside their image; clamp first.
                x0, y0 = max(0, int(x0)), max(0, int(y0))
                x1, y1 = min(im.width, int(x1)), min(im.height, int(y1))
                if c == FIGURE and len(figure_crops) < 1500 and (x1 - x0) > 60 and (y1 - y0) > 60:
                    crop = im.convert("RGB").crop((x0, y0, x1, y1))
                    crop.thumbnail((600, 600))
                    figure_crops.append(crop)
        print(f"  {shard.name}: real train so far {n_train} (skipped {n_skipped} test-set pages)", flush=True)

    # --- real val / test (Odia shard, IndicDLP's own held-out splits) ---
    for split, pattern in (("val", "validation-*.parquet"), ("test", "test-*.parquet")):
        n = 0
        for f in ODIA_SHARD.glob(pattern):
            for im, boxes in iter_parquet(f):
                save(im, split, f"real_odia_{n:05d}", boxes)
                n += 1
        print(f"real {split}: {n}")

    # --- synthetic (own labels mapped to IndicDLP + seal/stamp/signature) ---
    counts = {"train": 0, "val": 0, "test": 0}
    for sc in sorted(SYNTHETIC_LAYOUT.rglob("*.layout.json")):
        meta = json.loads(sc.read_text(encoding="utf-8"))
        split = bucket(meta["imageFilename"])
        im = Image.open(sc.with_name(meta["imageFilename"])).convert("RGB")
        gt = synthetic_labels(meta)

        pasted = []
        if split == "train" and figure_crops and rng.random() < 0.33:
            for _ in range(rng.randint(1, 3)):
                crop = rng.choice(figure_crops)
                s = rng.uniform(0.15, 0.4) * im.width / crop.width
                crop = crop.resize((max(8, int(crop.width * s)), max(8, int(crop.height * s))))
                x = rng.randint(0, max(0, im.width - crop.width))
                y = rng.randint(0, max(0, im.height - crop.height))
                im.paste(crop, (x, y))
                pasted.append((FIGURE, x, y, x + crop.width, y + crop.height))

        # A pasted figure hides whatever text it covers; drop those labels.
        gt = [b for b in gt if not any(_overlap(b[1:], p[1:]) > 0.5 for p in pasted)]
        save(im, split, f"syn_{Path(meta['imageFilename']).stem}", gt + pasted)
        counts[split] += 1
    print(f"synthetic: {counts}")

    (OUT / "data.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)), encoding="utf-8")
    print("wrote", OUT / "data.yaml")


def synthetic_labels(meta: dict) -> list[tuple]:
    out = []
    for r in meta["regions"]:
        if r["class"] in EXTRA_ID:
            out.append((EXTRA_ID[r["class"]], *r["bbox"]))
        elif r["class"] in SYNTHETIC_TO_INDICDLP:
            out.append((INDICDLP_NAMES.index(SYNTHETIC_TO_INDICDLP[r["class"]]), *r["bbox"]))
    return out


def _overlap(a, b) -> float:
    """Intersection over the smaller box's area."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])) or 1
    return ix * iy / small


if __name__ == "__main__":
    main()
