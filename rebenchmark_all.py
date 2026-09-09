"""
Fresh, apples-to-apples re-benchmark of all four wired-up OCR engines
(Tesseract, EasyOCR, PaddleOCR, TrOCR) against the full annotated dataset
(all three splits), using the exact production engine classes from
app/engines/ -- not a reimplementation.

Reports CER separately for real vs. synthetic fields per script, since the
charter already flagged that TrOCR's one clean score was likely leakage from
its synthetic/IIIT-HW-Dev-adjacent training data -- real-only numbers are the
ones that actually matter for picking a fine-tuning target.

Run from demo/ with its own venv:
  .venv\\Scripts\\python.exe rebenchmark_all.py
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import ANNOTATED_DIR, ANNOTATED_SPLITS
from app.engines.registry import get_engines
from app.imaging import bbox_crop, safe_cer
from PIL import Image

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def is_synthetic(record: dict) -> bool:
    return any("synthetic" in s.lower() for s in record.get("collectionSource", []))


def load_all_records():
    """Yield (script, split, image_path, record) for every field-level record."""
    for split in ANNOTATED_SPLITS:
        split_dir = ANNOTATED_DIR / split
        if not split_dir.is_dir():
            continue
        for script_dir in sorted(split_dir.iterdir()):
            if not script_dir.is_dir():
                continue
            script = script_dir.name
            for json_path in sorted(script_dir.glob("*.json")):
                with open(json_path, "r", encoding="utf-8") as f:
                    records = json.load(f)
                if not records:
                    continue
                image_path = script_dir / records[0]["imageFilename"]
                if not image_path.exists():
                    continue
                for rec in records:
                    yield script, split, image_path, rec


def main():
    engines = get_engines()
    print(f"Engines wired: {[e.name for e in engines]}")

    records = list(load_all_records())
    print(f"Loaded {len(records)} field-level records "
          f"across {len(set(s for s, *_ in records))} scripts, all splits")

    rows = []
    image_cache: dict[Path, Image.Image] = {}
    t0 = time.time()

    for i, (script, split, image_path, rec) in enumerate(records):
        gt = rec["groundTruth"]
        real = not is_synthetic(rec)

        if image_path not in image_cache:
            image_cache[image_path] = Image.open(image_path)
        im = image_cache[image_path]
        crop = bbox_crop(im, rec["boundingBox"]["vertices"])

        applicable = [e for e in engines if e.is_available_for(script)]
        for engine in applicable:
            hyp = engine.recognize(crop, script)
            cer = safe_cer(gt, hyp)
            rows.append({
                "script": script,
                "split": split,
                "real": real,
                "image": image_path.name,
                "engine": engine.name,
                "groundTruth": gt,
                "hyp": hyp,
                "cer": cer,
            })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  ...{i + 1}/{len(records)} records ({elapsed:.0f}s elapsed)")

    out_path = RESULTS_DIR / "rebenchmark_all_raw.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(rows)} rows to {out_path}")

    summarize(rows)


def mean(lst):
    return round(sum(lst) / len(lst), 4) if lst else None


def summarize(rows):
    # per script x engine x real/synthetic
    agg = defaultdict(list)
    for r in rows:
        if r["cer"] is None:
            continue
        agg[(r["script"], r["engine"], "real" if r["real"] else "synthetic")].append(r["cer"])

    lines = ["| Script | Engine | Real CER (n) | Synthetic CER (n) |",
             "|---|---|---|---|"]
    scripts = sorted(set(k[0] for k in agg))
    engine_names = sorted(set(k[1] for k in agg))
    for script in scripts:
        for engine in engine_names:
            real_vals = agg.get((script, engine, "real"), [])
            syn_vals = agg.get((script, engine, "synthetic"), [])
            if not real_vals and not syn_vals:
                continue
            real_s = f"{mean(real_vals)} ({len(real_vals)})" if real_vals else "-"
            syn_s = f"{mean(syn_vals)} ({len(syn_vals)})" if syn_vals else "-"
            lines.append(f"| {script} | {engine} | {real_s} | {syn_s} |")

    per_script_md = "\n".join(lines)
    print("\n" + per_script_md)

    # overall per-engine, REAL FIELDS ONLY (the number that matters for picking
    # a fine-tuning target -- synthetic/template data is not representative)
    overall_real = defaultdict(list)
    for r in rows:
        if r["cer"] is not None and r["real"]:
            overall_real[r["engine"]].append(r["cer"])

    lines2 = ["\n## Overall, REAL fields only (all scripts each engine supports)",
              "| Engine | Mean CER | N fields | Scripts covered |",
              "|---|---|---|---|"]
    for engine in sorted(overall_real):
        vals = overall_real[engine]
        covered = sorted(set(r["script"] for r in rows if r["engine"] == engine and r["real"]))
        lines2.append(f"| {engine} | {mean(vals)} | {len(vals)} | {', '.join(covered)} |")
    overall_md = "\n".join(lines2)
    print(overall_md)

    # head-to-head: only scripts >1 engine supports, real fields only -- the
    # fairest comparison since it controls for script difficulty
    contested_scripts = [s for s in scripts
                          if len({e for (sc, e, rs) in agg if sc == s}) > 1]
    lines3 = [f"\n## Head-to-head on contested scripts {contested_scripts} (real fields only)",
              "| Script | Engine | Mean CER | N |",
              "|---|---|---|---|"]
    for script in contested_scripts:
        for engine in engine_names:
            vals = agg.get((script, engine, "real"), [])
            if vals:
                lines3.append(f"| {script} | {engine} | {mean(vals)} | {len(vals)} |")
    contested_md = "\n".join(lines3)
    print(contested_md)

    md_path = RESULTS_DIR / "rebenchmark_all_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Fresh re-benchmark: all 4 engines, all splits, real vs synthetic\n\n")
        f.write(per_script_md + "\n")
        f.write(overall_md + "\n")
        f.write(contested_md + "\n")
    print(f"\nWrote summary to {md_path}")


if __name__ == "__main__":
    main()
