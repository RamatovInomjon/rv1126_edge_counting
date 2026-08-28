#!/usr/bin/env python3
"""
Carve a small train/val/test subset from all_crowdhuman for fast experimentation.

Source : $CROWDHUMAN_SRC/images/{train,val} + labels/{train,val}
Output : $CROWDHUMAN_SUBSET/{images,labels}/{train,val,test} + data.yaml   (symlinks)

Set CROWDHUMAN_SRC and CROWDHUMAN_SUBSET, or edit SRC/DST below.

train + test are sampled (disjoint) from the source *train* pool;
val is sampled from the source *val* pool. Only images that have a label file
are used. Deterministic via --seed.
"""
import argparse
import os
import random
from pathlib import Path

# Point these at your own CrowdHuman copy, or set the environment variables.
# SRC must contain images/<split> and labels/<split> in YOLO format.
SRC = Path(os.environ.get("CROWDHUMAN_SRC", "./crowdhuman/all"))
DST = Path(os.environ.get("CROWDHUMAN_SUBSET", "./crowdhuman_subset"))


def labelled_pairs(split):
    img_dir = SRC / "images" / split
    lbl_dir = SRC / "labels" / split
    pairs = []
    for img in img_dir.glob("*.jpg"):
        lbl = lbl_dir / (img.stem + ".txt")
        if lbl.exists():
            pairs.append((img, lbl))
    return pairs


def link(pairs, split):
    idir = DST / "images" / split
    ldir = DST / "labels" / split
    idir.mkdir(parents=True, exist_ok=True)
    ldir.mkdir(parents=True, exist_ok=True)
    for img, lbl in pairs:
        for src, dst in ((img, idir / img.name), (lbl, ldir / lbl.name)):
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src)   # absolute symlink
    print(f"  {split}: {len(pairs)} images -> {idir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=3000)
    ap.add_argument("--val", type=int, default=600)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    train_pool = labelled_pairs("train")
    val_pool = labelled_pairs("val")
    random.Random(a.seed).shuffle(train_pool)
    random.Random(a.seed + 1).shuffle(val_pool)

    need = a.train + a.test
    if len(train_pool) < need:
        raise SystemExit(f"train pool has {len(train_pool)} labelled imgs, need {need}")
    if len(val_pool) < a.val:
        raise SystemExit(f"val pool has {len(val_pool)} labelled imgs, need {a.val}")

    train_sel = train_pool[: a.train]
    test_sel = train_pool[a.train : a.train + a.test]   # disjoint from train
    val_sel = val_pool[: a.val]

    print(f"building subset at {DST}")
    link(train_sel, "train")
    link(val_sel, "val")
    link(test_sel, "test")

    yaml = DST / "data.yaml"
    yaml.write_text(
        f"path: {DST}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n  0: head\n  1: person\n"
    )
    print(f"wrote {yaml}")


if __name__ == "__main__":
    main()
