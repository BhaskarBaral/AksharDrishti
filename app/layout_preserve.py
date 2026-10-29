"""Layout preservation: combine the layout detector's regions with page-level
OCR lines into a structured document -- every region typed, in reading
order, carrying its own text (tables rebuilt as rows x columns) -- and render
it as Markdown. The router returns both, so the UI can redraw the page from
the structure and export it."""
import statistics

from PIL import Image

from app.engines.registry import engines_for_script
from app.layout import RTL_SCRIPTS, sort_reading_order
from app.layout_detector import detector

# IndicDLP's 42 classes + our 3, folded into the handful of roles a
# reconstructed document needs.
CLASS_GROUP = {
    **dict.fromkeys(["chapter-title", "headline", "sub-headline", "subsub-headline", "section-title",
                     "sub-section-title", "subsub-section-title"], "title"),
    **dict.fromkeys(["ordered-list", "unordered-list", "sub-ordered-list", "sub-unordered-list",
                     "subsub-ordered-list", "subsub-unordered-list", "options"], "list"),
    **dict.fromkeys(["table", "table-of-contents"], "table"),
    **dict.fromkeys(["figure-caption", "table-caption"], "caption"),
    **dict.fromkeys(["figure", "advertisement", "flag", "formula"], "figure"),
    **dict.fromkeys(["header", "footer", "page-number", "folio", "jumpline"], "page-furniture"),
    "placeholder-text": "form-field",
    "seal": "seal", "stamp": "stamp", "signature": "signature",
}
NON_TEXT_GROUPS = {"figure", "seal", "stamp", "signature"}
def group_of(cls: str) -> str:
    return CLASS_GROUP.get(cls, "text")


def page_engine(script: str):
    """Best page engine per script -- the OCR tab's measured printed-page
    ranking (ocr_router.PAGE_ENGINE_PREFERENCE), same default the /v1 API uses."""
    from app.routers.ocr_router import PAGE_ENGINE_PREFERENCE

    engines = {e.name: e for e in engines_for_script(script) if e.supports_page_level}
    order = PAGE_ENGINE_PREFERENCE.get(script, []) + list(engines)
    return next((engines[n] for n in order if n in engines), None)


def _xyxy(poly_or_box):
    if isinstance(poly_or_box[0], (list, tuple)):
        xs, ys = [p[0] for p in poly_or_box], [p[1] for p in poly_or_box]
        return [min(xs), min(ys), max(xs), max(ys)]
    return list(poly_or_box)


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b):
    return _area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def _table_grid(lines: list[dict], rtl: bool) -> list[list[str]]:
    """Rows by y-proximity, columns by clustering x-starts across the table."""
    if not lines:
        return []
    med_h = statistics.median(l["bbox"][3] - l["bbox"][1] for l in lines) or 1
    rows: list[list[dict]] = []
    for l in sorted(lines, key=lambda l: (l["bbox"][1] + l["bbox"][3]) / 2):
        yc = (l["bbox"][1] + l["bbox"][3]) / 2
        if rows and abs(yc - statistics.mean((m["bbox"][1] + m["bbox"][3]) / 2 for m in rows[-1])) <= med_h * 0.6:
            rows[-1].append(l)
        else:
            rows.append([l])
    starts = sorted(l["bbox"][0] for l in lines)
    cols = [starts[0]]
    for x in starts[1:]:
        if x - cols[-1] > med_h * 2:
            cols.append(x)
    grid = []
    for row in rows:
        cells = [""] * len(cols)
        for l in sorted(row, key=lambda l: l["bbox"][0]):
            ci = max(i for i, c in enumerate(cols) if c <= l["bbox"][0] + med_h)
            # OCR reads the table's ruling lines as "|"; they'd also break Markdown.
            cells[ci] = f"{cells[ci]} {l['text'].replace('|', ' ')}".strip()
            cells[ci] = " ".join(cells[ci].split())
        grid.append(cells[::-1] if rtl else cells)
    return grid


def analyze(image: Image.Image, script: str, threshold: float = 0.3) -> dict:
    image = image.convert("RGB")
    regions = detector.detect(image, threshold)
    engine = page_engine(script)
    lines = []
    if engine is not None:
        lines = [{"bbox": _xyxy(l["bbox"]), "text": l["text"], "confidence": l["confidence"]}
                 for l in engine.recognize_page(image, script) if l["text"].strip()]

    for i, r in enumerate(regions):
        r["id"] = f"r{i + 1:03d}"
        r["group"] = group_of(r["class"])
        r["lines"] = []

    # Each OCR line goes to the smallest text-bearing region covering most of it.
    text_regions = [r for r in regions if r["group"] not in NON_TEXT_GROUPS]
    unassigned = []
    for l in lines:
        la = _area(l["bbox"]) or 1
        hits = [r for r in text_regions if _inter(l["bbox"], r["bbox"]) / la >= 0.5]
        if hits:
            min(hits, key=lambda r: _area(r["bbox"]))["lines"].append(l)
        else:
            unassigned.append(l)
    # OCR text no detected region covers still belongs in the document.
    for l in unassigned:
        regions.append({"id": f"u{len(regions) + 1:03d}", "class": "unlabelled-text", "group": "text",
                        "bbox": l["bbox"], "score": None, "lines": [l]})

    rtl = script in RTL_SCRIPTS
    for r in regions:
        r["lines"] = sort_reading_order([{**l, "bbox": [[l["bbox"][0], l["bbox"][1]], [l["bbox"][2], l["bbox"][3]]]}
                                         for l in r["lines"]], rtl=rtl)
        r["lines"] = [{**l, "bbox": _xyxy(l["bbox"])} for l in r["lines"]]
        r["text"] = "\n".join(l["text"] for l in r["lines"])
        if r["group"] == "table":
            r["table"] = _table_grid(r["lines"], rtl)

    # Drop regions nested inside a same-group region (e.g. sub-list inside a list).
    ordered = sort_reading_order([{**r, "bbox": [[r["bbox"][0], r["bbox"][1]], [r["bbox"][2], r["bbox"][3]]]}
                                  for r in regions], rtl=rtl)
    ordered = [{**r, "bbox": _xyxy(r["bbox"])} for r in ordered]
    return {
        "width": image.width, "height": image.height, "script": script,
        "layoutModel": detector.model_name, "ocrEngine": engine.name if engine else None,
        "regions": ordered, "markdown": to_markdown(ordered),
    }


def to_markdown(regions: list[dict]) -> str:
    out = []
    for r in regions:
        g, text = r["group"], r.get("text", "").strip()
        if g == "figure":
            out.append(f"*[{r['class']}]*")
        elif g in ("seal", "stamp", "signature"):
            out.append(f"*[{g}]*")
        elif not text:
            continue
        elif g == "title":
            out.append(f"## {' '.join(text.splitlines())}")
        elif g == "list":
            out.append("\n".join(f"- {t}" for t in text.splitlines()))
        elif g == "table" and r.get("table"):
            grid = r["table"]
            width = max(len(row) for row in grid)
            rows = [row + [""] * (width - len(row)) for row in grid]
            md = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * width]
            md += ["| " + " | ".join(row) + " |" for row in rows[1:]]
            out.append("\n".join(md))
        elif g == "caption":
            out.append(f"*{' '.join(text.splitlines())}*")
        elif g == "form-field":
            out.append(f"{' '.join(text.splitlines())} ________")
        elif g == "page-furniture":
            out.append(f"<sub>{' '.join(text.splitlines())}</sub>")
        else:
            out.append(text)
    return "\n\n".join(out)
