"""mapping 37 style เดิม -> 5 กลุ่ม style (ไม่ผูกกับ type)

ergonomic ไม่อยู่ในรายการที่ผู้ใช้ให้มา -> ผู้ใช้เลือกให้อยู่ Modern & Minimalist (2026-11-10)
"High-tech" ในรายการ = "hightech" ใน csv
"""
import csv
from collections import OrderedDict
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent

GROUPS = OrderedDict([
    ("Modern & Minimalist", ["modern", "minimal", "minimalist", "scandinavian", "nordic",
                             "bauhaus", "streamline", "zen", "ergonomic"]),
    ("Classic & Ornamental", ["victorian", "classical", "classic", "baroque", "gothic",
                              "traditional", "colonial", "provincial", "deco"]),
    ("Industrial & Rustic", ["industrial", "brutalist", "rustic", "architectural", "primitive",
                             "deconstructed"]),
    ("Organic & Futuristic", ["organic", "biomorphic", "sculptural", "futuristic", "hightech",
                              "parametric", "postmodern"]),
    ("Cultural & Eclectic", ["bohemian", "eclectic", "retro", "japanese", "italian", "coastal"]),
])

STYLE_TO_GROUP = {s: g for g, styles in GROUPS.items() for s in styles}
assert len(STYLE_TO_GROUP) == sum(len(v) for v in GROUPS.values()), "style ซ้ำในหลายกลุ่ม"

TYPES = ["bed", "chair", "sofa", "table"]


def map_style(style):
    return STYLE_TO_GROUP.get(style.strip().lower())


def _check_train_covered():
    with open(DATA / "train.csv", encoding="utf-8", newline="") as f:
        styles = {r["style"] for r in csv.DictReader(f)}
    missing = sorted(s for s in styles if map_style(s) is None)
    if missing:
        raise RuntimeError(f"style ใน train.csv ที่ยังไม่มีกลุ่ม: {missing}")


_check_train_covered()
