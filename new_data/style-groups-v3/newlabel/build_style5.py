"""สร้าง manifest ฉบับ 5 กลุ่ม style ลง style5/ (ไม่แตะ manifest เดิม)

    python3 style5/build_style5.py

- train/validation/synthetic_test.csv : คอลัมน์เดิม + style_group
- real_source_test.csv               : คอลัมน์เดิม + style_group, orig_type, queue_reason, relabel_status
                                       (ใช้ผลจาก labeler/state.json ถ้ามี)
- style_groups.json, summary.csv
"""
import csv
import io
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style_groups import DATA, GROUPS, TYPES, map_style  # noqa: E402

OUT = DATA / "style5"
STATE = DATA / "labeler" / "state.json"
CLOSED_SPLITS = ["train", "validation", "synthetic_test"]

REASONS = {
    "missing": "label ว่าง",
    "type_oov": "type นอก 4 คลาส",
    "style_oov": "style นอก 37 style (ไม่มีกลุ่ม)",
}


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames), list(reader)


def write_atomic(path, text):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    os.replace(tmp, path)


def to_csv(fields, rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def queue_reason(row):
    t, s = row["type"].strip(), row["style"].strip()
    if not t or not s:
        return "missing"
    if t not in TYPES:
        return "type_oov"
    if map_style(s) is None:
        return "style_oov"
    return None


def load_state():
    if not STATE.exists():
        return {}
    with open(STATE, encoding="utf-8") as f:
        return json.load(f)


def status_of(label):
    if label is None:
        return "pending"
    if label.get("type") is None:
        return "unmappable"
    if label.get("style_group") is None:
        return "type_only"
    return "relabeled"


def build_real_source(state):
    """คืน (fields, rows) ของ real_source_test ฉบับ 5 กลุ่ม"""
    fields, rows = read_csv(DATA / "real_source_test.csv")
    fields = fields + ["style_group", "orig_type", "queue_reason", "relabel_status"]
    out = []
    for r in rows:
        o = dict(r, orig_type=r["type"], queue_reason="", relabel_status="kept")
        reason = queue_reason(r)
        if reason is None:
            o["style_group"] = map_style(r["style"])
        else:
            label = state.get(r["image_id"])
            st = status_of(label)
            o["queue_reason"], o["relabel_status"], o["style_group"] = reason, st, ""
            if st in ("relabeled", "type_only"):
                o["type"] = label["type"]
            if st == "relabeled":
                o["style_group"] = label["style_group"]
        out.append(o)
    return fields, out


def write_real_source(state):
    fields, rows = build_real_source(state)
    text = to_csv(fields, rows)
    write_atomic(OUT / "real_source_test.csv", text)
    return text, Counter(r["relabel_status"] for r in rows)


def summary_rows(split, rows):
    c = Counter((r["type"], r["style_group"]) for r in rows)
    return [{"split": split, "type": t, "style_group": g or "(ไม่มี)", "count": n}
            for (t, g), n in sorted(c.items(), key=lambda x: (x[0][0], x[0][1] or "~"))]


def main():
    OUT.mkdir(exist_ok=True)
    summary = []
    for split in CLOSED_SPLITS:
        fields, rows = read_csv(DATA / f"{split}.csv")
        for r in rows:
            g = map_style(r["style"])
            if g is None:
                raise RuntimeError(f"{split} {r['image_id']}: style '{r['style']}' ไม่มีกลุ่ม")
            r["style_group"] = g
        write_atomic(OUT / f"{split}.csv", to_csv(fields + ["style_group"], rows))
        summary += summary_rows(split, rows)
        print(f"{split}: {len(rows)} แถว")

    _, counts = write_real_source(load_state())
    _, rs_rows = build_real_source(load_state())
    summary += summary_rows("real_source_test", rs_rows)
    print(f"real_source_test: {len(rs_rows)} แถว {dict(counts)}")

    write_atomic(OUT / "summary.csv", to_csv(["split", "type", "style_group", "count"], summary))
    write_atomic(OUT / "style_groups.json", json.dumps(GROUPS, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
