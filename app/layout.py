"""Reading-order helper shared by engines that expose page-level (multi-line)
detection results, so the router doesn't need to know each engine's own
detection quirks."""
import statistics

# Scripts read right-to-left -- mirrors the private set in
# paddleocr_finetuned_engine.py, kept separate since that one only concerns
# single-crop CTC output reversal, not multi-line layout ordering.
RTL_SCRIPTS = {"urdu"}


def _y_center(item: dict) -> float:
    return sum(p[1] for p in item["bbox"]) / len(item["bbox"])


def _x_center(item: dict) -> float:
    return sum(p[0] for p in item["bbox"]) / len(item["bbox"])


def _height(item: dict) -> float:
    ys = [p[1] for p in item["bbox"]]
    return max(ys) - min(ys)


def sort_reading_order(items: list[dict], rtl: bool = False) -> list[dict]:
    """Group detected regions into horizontal lines (by y-center proximity)
    and order lines top-to-bottom, each line left-to-right (or right-to-left
    for RTL scripts). Not true layout analysis -- doesn't handle columns or
    tables, just enough to turn a bag of detections into a sane transcript."""
    if not items:
        return []

    by_y = sorted(items, key=_y_center)
    median_h = statistics.median(_height(it) for it in by_y) or 1.0

    lines: list[dict] = []
    for it in by_y:
        y = _y_center(it)
        line = next((l for l in lines if abs(y - l["y"]) <= median_h * 0.6), None)
        if line is None:
            lines.append({"y": y, "n": 1, "items": [it]})
        else:
            line["items"].append(it)
            line["y"] = (line["y"] * line["n"] + y) / (line["n"] + 1)
            line["n"] += 1

    lines.sort(key=lambda l: l["y"])
    ordered = []
    for line in lines:
        ordered.extend(sorted(line["items"], key=_x_center, reverse=rtl))
    return ordered
