import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

FIELDS = ["image_id", "image_path", "type", "style", "source_split", "pixel_hash", "dhash", "group_id"]


def rgb(image):
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        layer = image.convert("RGBA")
        bg = Image.new("RGBA", image.size, "white")
        return Image.alpha_composite(bg, layer).convert("RGB")
    return image.convert("RGB")


def image_hash(image):
    image = rgb(image)
    header = f"{image.width}x{image.height}:RGB:".encode()
    return hashlib.sha256(header + image.tobytes()).hexdigest()


def dhash(image):
    pixels = np.asarray(image.convert("L").resize((9, 8))).reshape(-1).tolist()
    bits = [pixels[y*9+x] > pixels[y*9+x+1] for y in range(8) for x in range(8)]
    return f"{sum(int(v) << i for i, v in enumerate(bits)):016x}"


def read_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def make_splits(rows, seed=42):
    """Group exact pixel duplicates; stratify joint type/style groups where possible."""
    groups = defaultdict(list)
    for row in rows:
        if not row.get("type") or not row.get("style"):
            raise ValueError("Every row needs non-empty type/style labels")
        groups[row["pixel_hash"]].append(row)
    strata = defaultdict(list)
    for group in groups.values():
        labels = {(r["type"], r["style"]) for r in group}
        if len(labels) != 1:
            raise ValueError("Identical pixels have conflicting labels; review before splitting")
        strata[next(iter(labels))].append(group)
    result = {"train": [], "val": [], "test": []}
    rng = random.Random(seed)
    rare = []
    for label, items in sorted(strata.items()):
        items.sort(key=lambda g: g[0]["pixel_hash"])
        rng.shuffle(items)
        n = len(items)
        holdout = max(1, round(n * .15)) if n >= 3 else 0
        if not holdout:
            rare.append({"type": label[0], "style": label[1], "groups": n})
        for i, group in enumerate(items):
            split = "val" if i < holdout else "test" if i < 2*holdout else "train"
            result[split].extend(group)
    if any(not result[s] for s in result):
        raise ValueError("Not enough label groups for non-empty train/val/test")
    return result, rare


def prepare(output, dataset_id, revision, limit=None):
    from .prepare import partition
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Output exists; use a new directory to preserve manifests")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if not isinstance(revision, str) or len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision.lower()):
        raise ValueError('revision must be a pinned 40-character commit SHA')
    from datasets import load_dataset
    output.mkdir(parents=True)
    image_dir = output / "images"
    image_dir.mkdir()
    rows, skipped = [], []
    # Streaming avoids loading all images into RAM; network downloads still required.
    source = load_dataset(dataset_id, revision=revision, streaming=True)
    for source_split, samples in source.items():
        for i, sample in enumerate(samples):
            if limit is not None and i >= limit:
                break
            image_id = f"{source_split}-{i:07d}"
            try:
                image = rgb(sample["image"])
                labels = {k: sample[k].strip().lower() for k in ("type", "style")}
                if not all(labels.values()):
                    raise ValueError("empty label")
                path = image_dir / f"{image_id}.png"
                image.save(path)
                rows.append({"image_id": image_id, "image_path": str(path.relative_to(output)),
                             **labels, "source_split": source_split,
                             "pixel_hash": image_hash(image), "dhash": dhash(image),
                             "_split": "test" if source_split.lower() == "test" else ""})
            except (ValueError, OSError, KeyError, AttributeError) as error:
                skipped.append({"image_id": image_id, "reason": str(error)})
    splits, rare = partition(rows, {'seed':42, 'val_fraction':.15, 'test_fraction':.15})
    for split, values in splits.items():
        write_rows(output / f"{split}.csv", values)
    audit = {"dataset": dataset_id, "revision": revision, "limited_per_source_split": limit,
             "split_policy": "joint_label_exact_duplicate_groups_70_15_15_approx_preserve_source_test",
             "seed": 42, "rows": len(rows), "skipped": skipped, "rare_train_only": rare,
             "counts": {k: len(v) for k, v in splits.items()},
             "labels": {key: dict(Counter(r[key] for r in rows)) for key in ("type", "style")},
             "near_duplicate_review": "REQUIRED: dhash is provided; exact grouping alone is insufficient"}
    (output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def validate_splits(root):
    root = Path(root)
    splits = {s: read_rows(root / f"{s}.csv") for s in ("train", "val", "test")}
    hashes, ids, group_splits, labels_by_hash = {}, set(), {}, {}
    for split, rows in splits.items():
        if not rows:
            raise ValueError(f"Empty {split} split")
        for row in rows:
            if row.get('source_split', '').lower() == 'test' and split != 'test':
                raise ValueError('Source test row in training or validation; rebuild the dataset snapshot')
            if row["image_id"] in ids:
                raise ValueError("Duplicate image_id in manifests")
            ids.add(row["image_id"])
            with Image.open(root / row["image_path"]) as im:
                if image_hash(im) != row["pixel_hash"]:
                    raise ValueError(f"Changed image: {row['image_id']}")
            if not row.get("type") or not row.get("style"):
                raise ValueError("Labels must be non-empty")
            labels = (row["type"], row["style"])
            if labels_by_hash.setdefault(row["pixel_hash"], labels) != labels:
                raise ValueError("Identical pixels have conflicting labels")
            group = row.get("group_id")
            if group and group_splits.setdefault(group, split) != split:
                raise ValueError("Source/product group crosses splits")
            old = hashes.setdefault(row["pixel_hash"], split)
            if old != split:
                raise ValueError("Exact image duplicate crosses splits")
    mapping = {key: {v: i for i, v in enumerate(sorted({r[key] for r in splits['train']}))}
               for key in ("type", "style")}
    for split, rows in splits.items():
        for row in rows:
            for key in mapping:
                if row[key] not in mapping[key]:
                    raise ValueError(f"Unseen {key} label in {split}: {row[key]}")
    return splits, mapping


class Square:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        return ImageOps.pad(rgb(image), (self.size, self.size), color="white")


def transform(size, augment=False):
    steps = [Square(size)]
    from .augmentation import pil_transforms
    steps += pil_transforms(augment)
    return T.Compose(steps + [T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])


class FurnitureDataset(Dataset):
    def __init__(self, rows, root, mapping, size=224, augment=False):
        self.rows, self.root, self.mapping = rows, Path(root), mapping
        self.transform = transform(size, augment)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["image_path"]) as image:
            tensor = self.transform(image)
        return tensor, {key: torch.tensor(self.mapping[key][row[key]]) for key in self.mapping}
