#!/usr/bin/env python3
# Detection-based counting metrics. Tune confidence tau on VAL (no test leakage),
# report MAE/RMSE/bias/NAE on TEST for FP32 and on-device INT8.
import json
import os, sys
import numpy as np
NAMES = ["head", "person"]

def gt_counts(labels_dir, stems):
    out = {}
    for s in stems:
        f = os.path.join(labels_dir, s + ".txt"); c = [0, 0]
        if os.path.exists(f):
            for ln in open(f):
                p = ln.split()
                if len(p) >= 5: c[int(float(p[0]))] += 1
        out[s] = c
    return out

def counts_at(preds, tau):
    out = {}
    for s, dets in preds.items():
        c = [0, 0]
        for d in dets:
            if d[4] >= tau: c[int(d[5])] += 1
        out[s] = c
    return out

def target_val(c, target):
    return sum(c) if target == "total" else c[NAMES.index(target)]

def metrics(pcounts, gcounts, target):
    e, g = [], []
    for s in gcounts:
        p = target_val(pcounts.get(s, [0, 0]), target); gg = target_val(gcounts[s], target)
        e.append(p - gg); g.append(gg)
    e = np.array(e, float); g = np.array(g, float)
    mae = np.mean(np.abs(e)); rmse = np.sqrt(np.mean(e**2)); bias = np.mean(e)
    nae = np.mean(np.abs(e) / np.maximum(g, 1))
    return dict(MAE=mae, RMSE=rmse, bias=bias, NAE=nae, gt_mean=g.mean())

def tune(preds_val, gt_val, target, taus):
    best_t, best_mae = None, 1e9
    for t in taus:
        m = metrics(counts_at(preds_val, t), gt_val, target)["MAE"]
        if m < best_mae: best_mae, best_t = m, t
    return best_t

def main():
    cfgs = ["yolov8n_320", "yolov8n_448", "yolov8n_640", "yolov8s_320"]
    val_lbl = os.environ.get("CROWDHUMAN_SUBSET", "./crowdhuman_subset") + "/labels/val"
    test_lbl = "device_test/labels"
    taus = np.round(np.arange(0.05, 0.90, 0.025), 3)
    rows = []
    for cfg in cfgs:
        pv = json.load(open(f"preds_fp32_val_{cfg}.json"))
        pt_fp = json.load(open(f"preds_fp32_{cfg}.json"))
        pt_i8 = json.load(open(f"preds_int8_{cfg}.json"))
        gv = gt_counts(val_lbl, list(pv.keys()))
        gt = gt_counts(test_lbl, list(pt_fp.keys()))
        for target in ["head", "person", "total"]:
            tau = tune(pv, gv, target, taus)
            mfp = metrics(counts_at(pt_fp, tau), gt, target)
            mi8 = metrics(counts_at(pt_i8, tau), gt, target)
            rows.append(dict(config=cfg, target=target, tau=tau,
                gt_mean=round(mfp["gt_mean"], 1),
                fp32_MAE=round(mfp["MAE"], 2), fp32_RMSE=round(mfp["RMSE"], 2), fp32_bias=round(mfp["bias"], 2),
                int8_MAE=round(mi8["MAE"], 2), int8_RMSE=round(mi8["RMSE"], 2), int8_bias=round(mi8["bias"], 2),
                int8_NAE=round(mi8["NAE"], 3)))
    import csv
    with open("../results/counting_errors.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # pretty print
    print("%-13s %-7s %5s %7s | %8s %8s %8s | %8s %8s %8s" % (
        "config","target","tau","gt_mean","fp32_MAE","fp32_RMSE","fp32_bias","int8_MAE","int8_RMSE","int8_bias"))
    for r in rows:
        print("%-13s %-7s %5.3f %7.1f | %8.2f %8.2f %+8.2f | %8.2f %8.2f %+8.2f" % (
            r["config"],r["target"],r["tau"],r["gt_mean"],
            r["fp32_MAE"],r["fp32_RMSE"],r["fp32_bias"],r["int8_MAE"],r["int8_RMSE"],r["int8_bias"]))
    print("\nwrote ../results/counting_errors.csv")

if __name__ == "__main__":
    main()
