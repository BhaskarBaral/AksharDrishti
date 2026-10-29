"""
Compare the untouched IndicDLP checkpoint with our fine-tune.

  real test   IndicDLP's own held-out Odia test split (real pages, real
              labels, 42 classes) -- did the fine-tune keep real-page skill?
  synthetic   held-out synthetic-layout pages -- seal / stamp / signature.
  complex     the dataset repo's 325 real complex-testset pages (unlabelled):
              how many seal / stamp / signature / figure boxes each model
              fires, a proxy for false positives on real forms.

Writes training/layout/runs/eval.json.

Usage:
  .venv\\Scripts\\python.exe training/layout/evaluate_yolo.py
"""
import json
from collections import Counter
from pathlib import Path

from doclayout_yolo import YOLOv10

HERE = Path(__file__).resolve().parent
YOLO = HERE / "yolo"
BASE = Path(r"C:\Users\Legion\Desktop\AksharDristhi\datasets\external\indicdlp-detector-zenodo"
            r"\doclayout_yolo_indicdlp.pt")
FINETUNED = HERE / "runs" / "indicdlp_ft" / "weights" / "best.pt"
COMPLEX = Path(r"C:\Users\Legion\Desktop\Akshar_dristhii_code\complex-testset\pages")


def names_from_yaml() -> list[str]:
    lines = (YOLO / "data.yaml").read_text(encoding="utf-8").split("names:")[1].strip().splitlines()
    return [l.split(":", 1)[1].strip() for l in lines]


def write_yaml(name: str, names: list[str], test_list: Path) -> Path:
    p = YOLO / name
    p.write_text(f"path: {YOLO.as_posix()}\ntrain: images/train\nval: {test_list.as_posix()}\n"
                 "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)), encoding="utf-8")
    return p


def per_class(metrics, names) -> dict:
    return {names[int(c)]: round(float(ap), 4) for c, ap in zip(metrics.box.ap_class_index, metrics.box.ap50)}


def main():
    names = names_from_yaml()
    test_imgs = sorted((YOLO / "images" / "test").glob("*.jpg"))
    real_list, syn_list = YOLO / "test_real.txt", YOLO / "test_syn.txt"
    real_list.write_text("\n".join(p.as_posix() for p in test_imgs if p.stem.startswith("real_")))
    syn_list.write_text("\n".join(p.as_posix() for p in test_imgs if p.stem.startswith("syn_")))

    out = {}
    # Each at the size it was trained at (train_yolo.py fine-tunes at 800).
    for label, weights, nm, imgsz in (("base", BASE, names[:42], 1024), ("finetuned", FINETUNED, names, 800)):
        if not weights.exists():
            continue
        model = YOLOv10(str(weights))
        r = {}
        m = model.val(data=str(write_yaml(f"eval_real_{label}.yaml", nm, real_list)), split="val", imgsz=imgsz,
                      batch=4, device=0, plots=False, verbose=False)
        r["real_test_map50"], r["real_test_map50_95"] = round(m.box.map50, 4), round(m.box.map, 4)
        r["real_test_ap50"] = per_class(m, nm)
        if label == "finetuned":
            m = model.val(data=str(write_yaml("eval_syn.yaml", nm, syn_list)), split="val", imgsz=imgsz,
                          batch=4, device=0, plots=False, verbose=False)
            r["synthetic_test_ap50"] = {k: v for k, v in per_class(m, nm).items() if k in ("seal", "stamp", "signature")}
        fires = Counter()
        for p in sorted(COMPLEX.glob("*.png")):
            res = model.predict(str(p), imgsz=imgsz, conf=0.5, device=0, verbose=False)[0]
            fires.update(res.names[int(c)] for c in res.boxes.cls.tolist())
        r["complex_testset_detections_conf50"] = {k: fires[k] for k in ("seal", "stamp", "signature", "figure")}
        out[label] = r
        print(label, json.dumps({k: v for k, v in r.items() if k != "real_test_ap50"}, indent=1), flush=True)
    (HERE / "runs").mkdir(exist_ok=True)
    (HERE / "runs" / "eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
