"""
Build TestData/ -- the demo's held-out test set, one folder per language plus
a separate layout/ folder:

  TestData/
    <script>/word_level/   real handwritten word crops + manifest.json
    <script>/page_level/   handwritten pages + manifest.json
                           (real where a real source exists, else synthetic
                           handwriting-style pages -- every entry says which)
    layout/real_labelled/      IndicDLP Odia test split, real region labels
    layout/real_unlabelled/    real forms from the dataset repo's complex-testset
    layout/synthetic_labelled/ held-out synthetic-layout pages (seals/stamps/...)

Everything is copied read-only from the dataset repo and is held out from
every model the demo uses:
  * word crops: none listed in demo/training/datasets/<script>/{train,val}.txt
    (the fine-tuned recognizers' training data).
  * synthetic pages: none whose fields were cropped into any synthetic pilot
    train.txt, and only the synthetic-layout pages in layout_to_coco.py's
    "test" hash bucket (never seen by the layout fine-tune).
  * IndicDLP Odia pages: IndicDLP's own test split (the layout fine-tune only
    trained on IndicDLP train shards).

Usage:
  python TestData/_tools/build_testdata.py
"""
import hashlib
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = Path(r"C:\Users\Legion\Desktop\Akshar_dristhii_code")
DEMO = ROOT.parent / "demo"
TRAIN_LISTS = DEMO / "training" / "datasets"
LAYOUT_YOLO = DEMO / "training" / "layout" / "yolo"

SCRIPTS = ["devanagari", "bengali", "tamil", "telugu", "urdu", "gujarati", "kannada", "malayalam", "odia",
           "punjabi"]
WORDS_PER_SCRIPT = 50
PAGES_PER_SCRIPT = 10
IMAGE_EXTS = (".jpeg", ".jpg", ".png")
LAYOUT_REAL_LABELLED = 20
# Picked by eye from complex-testset/pages: worksheets with drawings, dense
# bank forms, tables, a seal-like box.
LAYOUT_REAL_UNLABELLED = [
    ("candidate_007.png", "bengali"), ("candidate_033.png", "bengali"), ("candidate_054.png", "bengali"),
    ("candidate_017.png", "devanagari"), ("candidate_073.png", "bengali"), ("candidate_153.png", "bengali"),
    ("indicdlp_odia_fm_or_000247_0.png", "odia"), ("indicdlp_odia_fm_or_000020_0.png", "odia"),
]
SYNTH_LAYOUT_DOCTYPES = ["kv", "table", "bikv", "mixed"]


def image_for(json_path: Path) -> Path | None:
    return next((json_path.with_suffix(e) for e in IMAGE_EXTS if json_path.with_suffix(e).exists()), None)


def load_records(json_path: Path) -> list[dict]:
    d = json.loads(json_path.read_text(encoding="utf-8"))
    return d if isinstance(d, list) else [d]


def trained_on(script: str) -> tuple[set[str], set[str]]:
    """-> (real word image filenames, synthetic source page stems) used in training."""
    words, pages = set(), set()
    for name in ("train.txt", "val.txt"):
        p = TRAIN_LISTS / script / name
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            fname = line.split("\t")[0].replace("\\", "/").rsplit("/", 1)[-1]
            if "synthetic_crops" in line:
                pages.add(fname.rsplit(".", 1)[0].rsplit("_", 1)[0])
            else:
                words.add(fname)
    return words, pages


def word_level(script: str, used_words: set[str], rng: random.Random) -> list[dict]:
    out_dir = ROOT / script / "word_level"
    # Human-reviewed held-out words first, then the rest of the unused real pool.
    reviewed = {p.stem for split in ("test", "val", "train")
                for p in (DATASET / "annotated" / split / script).glob("*_real_*.json")}
    pool = []
    for folder in ("wordlevel", "normalized"):
        for jp in sorted((DATASET / "raw-data" / script / folder).glob("*.json")):
            img = image_for(jp)
            if img is None or img.name in used_words or "pagehw" in jp.stem or "indicdlp" in jp.stem:
                continue
            rec = load_records(jp)[0]
            if rec.get("imageTextType") != "handwritten-text":
                continue
            pool.append((jp.stem not in reviewed, rng.random(), jp, img, rec))
    pool.sort(key=lambda t: (t[0], t[1]))
    manifest = []
    for _, _, jp, img, rec in pool[:WORDS_PER_SCRIPT]:
        shutil.copy2(img, out_dir / img.name)
        manifest.append({"file": img.name, "groundTruth": rec["groundTruth"], "source": "real",
                         "humanReviewed": jp.stem in reviewed, "collectionSource": rec.get("collectionSource"),
                         "license": rec.get("license")})
    return manifest


def page_level(script: str, used_pages: set[str], rng: random.Random) -> list[dict]:
    out_dir = ROOT / script / "page_level"
    real = sorted((DATASET / "raw-data" / script / "pagelevel").glob("*_pagehw_*.json"))
    manifest = []
    for jp in rng.sample(real, min(PAGES_PER_SCRIPT, len(real))):
        img, rec = image_for(jp), load_records(jp)[0]
        shutil.copy2(img, out_dir / f"real_{img.name}")
        manifest.append({"file": f"real_{img.name}", "groundTruth": rec["groundTruth"], "source": "real",
                         "collectionSource": rec.get("collectionSource"), "license": rec.get("license")})
    if manifest:
        return manifest
    # No real handwritten page source for this script: synthetic
    # handwriting-style form pages (hw doctype), clean + moderate tiers.
    for tier in ("clean", "moderate"):
        cands = [jp for jp in sorted((DATASET / "synthetic" / script / "hw" / tier).glob("*.json"))
                 if not jp.name.endswith(".layout.json") and jp.stem not in used_pages]
        for jp in rng.sample(cands, min(PAGES_PER_SCRIPT // 2, len(cands))):
            img, recs = image_for(jp), load_records(jp)
            shutil.copy2(img, out_dir / f"synthetic_{img.name}")
            manifest.append({
                "file": f"synthetic_{img.name}", "source": "synthetic", "quality": tier,
                "groundTruth": "\n".join(r["groundTruth"] for r in recs),
                "lines": [{"groundTruth": r["groundTruth"], "boundingBox": r.get("boundingBox")} for r in recs],
            })
    return manifest


def bucket(name: str) -> str:
    v = int(hashlib.md5(name.encode()).hexdigest(), 16) % 10
    return "train" if v < 8 else ("val" if v == 8 else "test")


def layout(rng: random.Random) -> dict:
    counts = {}
    # Real, labelled: IndicDLP's own Odia test split as written by prepare_yolo.py.
    out = ROOT / "layout" / "real_labelled"
    names = {}
    yaml = LAYOUT_YOLO / "data.yaml"
    if yaml.exists():
        lines = yaml.read_text(encoding="utf-8").split("names:")[1].strip().splitlines()
        names = {int(l.split(":")[0]): l.split(":", 1)[1].strip() for l in lines}
    imgs = sorted((LAYOUT_YOLO / "images" / "test").glob("real_odia_*.jpg"))
    manifest = []
    from PIL import Image
    for img in rng.sample(imgs, min(LAYOUT_REAL_LABELLED, len(imgs))):
        w, h = Image.open(img).size
        regions = []
        for line in (LAYOUT_YOLO / "labels" / "test" / f"{img.stem}.txt").read_text().splitlines():
            c, cx, cy, bw, bh = line.split()
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            regions.append({"class": names.get(int(c), c),
                            "bbox": [round(cx - bw / 2, 1), round(cy - bh / 2, 1), round(cx + bw / 2, 1), round(cy + bh / 2, 1)]})
        shutil.copy2(img, out / img.name)
        manifest.append({"file": img.name, "script": "odia", "source": "real",
                         "collectionSource": "IndicDLP (ai4bharat), test split -- MIT", "regions": regions})
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    counts["real_labelled"] = len(manifest)

    # Real, unlabelled: hand-picked complex real forms.
    out = ROOT / "layout" / "real_unlabelled"
    manifest = []
    for name, script in LAYOUT_REAL_UNLABELLED:
        src = DATASET / "complex-testset" / "pages" / name
        if src.exists():
            shutil.copy2(src, out / name)
            manifest.append({"file": name, "script": script, "source": "real",
                             "collectionSource": "IndicDLP Forms pages via complex-testset -- MIT", "regions": None})
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    counts["real_unlabelled"] = len(manifest)

    # Synthetic, labelled: one held-out page per script x doctype.
    out = ROOT / "layout" / "synthetic_labelled"
    manifest = []
    for script in SCRIPTS:
        for doctype in SYNTH_LAYOUT_DOCTYPES:
            for sc in sorted((DATASET / "synthetic-layout" / script / doctype).rglob("*.layout.json")):
                meta = json.loads(sc.read_text(encoding="utf-8"))
                if bucket(meta["imageFilename"]) != "test":
                    continue
                shutil.copy2(sc.with_name(meta["imageFilename"]), out / meta["imageFilename"])
                manifest.append({"file": meta["imageFilename"], "script": script, "doctype": doctype,
                                 "quality": meta["quality"], "source": "synthetic",
                                 "regions": [{"class": r["class"], "bbox": r["bbox"]} for r in meta["regions"]]})
                break
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    counts["synthetic_labelled"] = len(manifest)
    return counts


def main():
    rng = random.Random(42)
    summary = {}
    for script in SCRIPTS:
        for level in ("word_level", "page_level"):
            d = ROOT / script / level
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True)
        used_words, used_pages = trained_on(script)
        words = word_level(script, used_words, rng)
        pages = page_level(script, used_pages, rng)
        for level, m in (("word_level", words), ("page_level", pages)):
            (ROOT / script / level / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1),
                                                                encoding="utf-8")
        summary[script] = {"word_level": len(words), "word_level_human_reviewed": sum(w["humanReviewed"] for w in words),
                           "page_level": len(pages), "page_level_source": sorted({p["source"] for p in pages})}
        print(script, summary[script], flush=True)
    for sub in ("real_labelled", "real_unlabelled", "synthetic_labelled"):
        d = ROOT / "layout" / sub
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    summary["layout"] = layout(rng)
    print("layout", summary["layout"])
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
