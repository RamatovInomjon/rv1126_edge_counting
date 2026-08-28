#!/usr/bin/env python3
"""
Train a matrix of YOLOv8 (n/s) x input-size configs on the crowdhuman_subset,
then evaluate each on the held-out test split and tabulate accuracy + model size.

Default matrix (per the RV1126 edge study):
    yolov8n @ 320, yolov8n @ 448, yolov8n @ 640, yolov8s @ 320

Speed is measured separately on the camera (see bench_device.py) after the
best.pt models are exported to RKNN — this script fills everything except
device FPS, which the bench step appends.

Run:
    conda activate yolo
    python train_matrix.py                       # full matrix
    python train_matrix.py --epochs 1            # smoke test
    python train_matrix.py --configs yolov8n:640,yolov8s:640
"""
import argparse
import csv
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "crowdhuman_subset" / "data.yaml"
RUNS = HERE / "runs"
CSV = HERE / "results.csv"

DEFAULT_BATCH = 64        # default batch; per-config override via model:imgsz:batch
# (model, imgsz, batch-or-None). s is a bigger model -> smaller batch.
DEFAULT_MATRIX = [("yolov8n", 320, 64), ("yolov8n", 448, 64),
                  ("yolov8n", 640, 64), ("yolov8s", 320, 32)]


def parse_configs(s):
    out = []
    for tok in s.split(","):
        parts = tok.split(":")
        name = parts[0].strip()
        sz = int(parts[1])
        b = int(parts[2]) if len(parts) > 2 else None   # optional per-config batch
        out.append((name, sz, b))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=parse_configs, default=DEFAULT_MATRIX,
                    help='e.g. "yolov8n:320,yolov8n:640,yolov8s:320"')
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="batch for every config")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--data", default=str(DATA))
    args = ap.parse_args()

    from ultralytics import YOLO
    import torch

    rows = []
    print(f"data={args.data}  configs={args.configs}  epochs={args.epochs}\n")

    for model_name, imgsz, cbatch in args.configs:
        batch = cbatch if cbatch else args.batch
        name = f"{model_name}_{imgsz}"
        print(f"\n===== training {name}  (imgsz={imgsz} batch={batch}) =====")
        t0 = time.time()

        model = YOLO(f"{model_name}.pt")
        model.train(
            data=args.data, epochs=args.epochs, imgsz=imgsz, batch=batch,
            device=args.device, workers=args.workers, patience=args.patience,
            project=str(RUNS), name=name, exist_ok=True,
            cos_lr=True, close_mosaic=10, amp=True, val=True, plots=False, verbose=False,
        )
        train_s = time.time() - t0
        save_dir = Path(model.trainer.save_dir)
        best = save_dir / "weights" / "best.pt"

        # evaluate the best weights on the held-out TEST split
        m = YOLO(best)
        metrics = m.val(data=args.data, split="test", imgsz=imgsz,
                        batch=batch, device=args.device, verbose=False, plots=False)
        params_m = sum(p.numel() for p in m.model.parameters()) / 1e6
        size_mb = best.stat().st_size / 1e6

        row = {
            "config": name, "model": model_name, "imgsz": imgsz,
            "params_M": round(params_m, 2), "best_pt_MB": round(size_mb, 2),
            "test_mAP50": round(float(metrics.box.map50), 4),
            "test_mAP50_95": round(float(metrics.box.map), 4),
            "mAP50_head": round(float(metrics.box.maps[0]), 4) if len(metrics.box.maps) > 0 else None,
            "mAP50_person": round(float(metrics.box.maps[1]), 4) if len(metrics.box.maps) > 1 else None,
            "train_min": round(train_s / 60, 1),
            "device_fps": "",   # filled by bench_device.py
            "best_pt": str(best),
        }
        rows.append(row)
        # write incrementally so partial results survive an interrupt
        with open(CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"  {name}: mAP50={row['test_mAP50']} mAP50-95={row['test_mAP50_95']} "
              f"params={row['params_M']}M size={row['best_pt_MB']}MB {row['train_min']}min")

    print("\n================ ACCURACY SUMMARY ================")
    hdr = ["config", "params_M", "best_pt_MB", "test_mAP50", "test_mAP50_95", "train_min"]
    print("  ".join(f"{h:>12}" for h in hdr))
    for r in rows:
        print("  ".join(f"{str(r[h]):>12}" for h in hdr))
    print(f"\nfull table -> {CSV}")
    print("next: export best.pt -> ONNX -> RKNN, then `python bench_device.py` for on-device FPS")


if __name__ == "__main__":
    main()
