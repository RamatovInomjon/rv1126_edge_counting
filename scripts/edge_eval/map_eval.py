#!/usr/bin/env python3
# COCO-style mAP@0.5 and mAP@0.5:0.95 from a predictions.json + YOLO label dir.
# preds.json: { stem: [[x1,y1,x2,y2,score,cls], ...] }  (normalized xyxy)
# labels:     <stem>.txt lines "cls xc yc w h"          (normalized)
import os, sys, json, glob
import numpy as np

NC = 2; NAMES = ["head", "person"]

def load_gt(labels_dir, stems):
    gt = {}
    for s in stems:
        f = os.path.join(labels_dir, s + ".txt")
        boxes = []
        if os.path.exists(f):
            for ln in open(f):
                p = ln.split()
                if len(p) < 5: continue
                c, xc, yc, w, h = int(float(p[0])), *map(float, p[1:5])
                boxes.append([c, xc - w/2, yc - h/2, xc + w/2, yc + h/2])
        gt[s] = np.array(boxes) if boxes else np.zeros((0, 5))
    return gt

def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0: return np.zeros((len(a), len(b)))
    area_a = (a[:, 2]-a[:, 0])*(a[:, 3]-a[:, 1])
    area_b = (b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1])
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2-x1, 0, None) * np.clip(y2-y1, 0, None)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)

def ap_101(rec, prec):
    # COCO 101-point interpolation
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre)-1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    x = np.linspace(0, 1, 101)
    return np.trapz(np.interp(x, mrec, mpre), x) if False else np.mean(np.interp(x, mrec, mpre))

def evaluate(preds, gt, iou_thr):
    # returns per-class AP at this IoU threshold
    aps = []
    for c in range(NC):
        # gather predictions for class c
        P = []  # (score, stem, box)
        npos = 0
        for s in gt:
            g = gt[s]; npos += int((g[:, 0] == c).sum())
        for s, dets in preds.items():
            for d in dets:
                if int(d[5]) == c:
                    P.append((d[4], s, np.array(d[:4])))
        if npos == 0:
            aps.append(float("nan")); continue
        P.sort(key=lambda z: -z[0])
        tp = np.zeros(len(P)); fp = np.zeros(len(P))
        used = {s: np.zeros(((gt[s][:, 0] == c).sum(),)) for s in gt}
        gtc = {s: gt[s][gt[s][:, 0] == c][:, 1:] for s in gt}
        for i, (sc, s, box) in enumerate(P):
            G = gtc.get(s, np.zeros((0, 4)))
            if len(G) == 0: fp[i] = 1; continue
            ious = iou_matrix(box[None], G)[0]
            j = ious.argmax()
            if ious[j] >= iou_thr and used[s][j] == 0:
                tp[i] = 1; used[s][j] = 1
            else:
                fp[i] = 1
        tpc = np.cumsum(tp); fpc = np.cumsum(fp)
        rec = tpc / (npos + 1e-9); prec = tpc / (tpc + fpc + 1e-9)
        aps.append(ap_101(rec, prec))
    return aps

def main():
    preds_path, labels_dir = sys.argv[1], sys.argv[2]
    preds = json.load(open(preds_path))
    gt = load_gt(labels_dir, list(preds.keys()))
    thrs = np.arange(0.5, 1.0, 0.05)
    per_thr = np.array([evaluate(preds, gt, t) for t in thrs])   # [10, NC]
    ap50 = per_thr[0]                    # per class @0.5
    ap5095 = np.nanmean(per_thr, axis=0) # per class mean over thr
    m50 = np.nanmean(ap50); m5095 = np.nanmean(ap5095)
    print("mAP@0.5      = %.4f" % m50)
    print("mAP@0.5:0.95 = %.4f" % m5095)
    for c in range(NC):
        print("  %-7s AP50=%.4f  AP50-95=%.4f" % (NAMES[c], ap50[c], ap5095[c]))
    # emit machine-readable line
    print("RESULT %.4f %.4f %.4f %.4f" % (m50, m5095, ap50[0], ap50[1]))

if __name__ == "__main__":
    main()
