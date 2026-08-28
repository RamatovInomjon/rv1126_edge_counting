#!/usr/bin/env python3
"""Generate all result figures for the RV1126 head/person counting study."""
import csv, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path("results/figures"); OUT.mkdir(parents=True, exist_ok=True)
C = {"n": "#1f77b4", "s": "#d62728", "py": "#9467bd", "cpp": "#2ca02c", "fused": "#ff7f0e",
     "fp32": "#1f77b4", "int8": "#ff7f0e", "pre": "#8c9eff", "inf": "#4caf50", "post": "#e53935"}
CFG = ["yolov8n_320", "yolov8n_448", "yolov8n_640", "yolov8s_320"]
LAB = ["n@320", "n@448", "n@640", "s@320"]

def rd(p):
    return list(csv.DictReader(open(p)))

master = {r["config"]: r for r in rd("results/results_master.csv")}
acc = {r["config"]: r for r in rd("results/accuracy_fp32_vs_int8.csv")}
opt = rd("results/optimized_pipeline.csv")
cnt = rd("results/counting_errors.csv")

# 1) Accuracy vs input resolution (YOLOv8n)
plt.figure(figsize=(6, 4.2))
res = [320, 448, 640]
m50 = [float(master[f"yolov8n_{r}"]["test_mAP50"]) for r in res]
m5095 = [float(master[f"yolov8n_{r}"]["test_mAP50_95"]) for r in res]
plt.plot(res, m50, "o-", color=C["n"], lw=2, label="mAP@0.5")
plt.plot(res, m5095, "s--", color="#17becf", lw=2, label="mAP@0.5:0.95")
s50 = float(master["yolov8s_320"]["test_mAP50"])
plt.scatter([320], [s50], color=C["s"], s=90, marker="D", zorder=5, label="YOLOv8s@320")
for r, y in zip(res, m50): plt.annotate(f"{y:.3f}", (r, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
plt.xlabel("Input resolution (px)"); plt.ylabel("test mAP"); plt.title("Accuracy vs. input resolution (YOLOv8n)")
plt.xticks(res); plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
plt.savefig(OUT / "fig1_accuracy_vs_resolution.png", dpi=150); plt.close()

# 2) Pareto: accuracy vs NPU throughput
plt.figure(figsize=(6.4, 4.4))
for cfg, lab in zip(CFG, LAB):
    x = float(master[cfg]["npu_inf_s"]); y = float(master[cfg]["test_mAP50"])
    n = cfg.startswith("yolov8n")
    plt.scatter(x, y, s=150, c=C["n"] if n else C["s"], marker="o" if n else "s",
                edgecolor="k", zorder=3)
    plt.annotate(f"{lab}\n{y:.3f}", (x, y), textcoords="offset points", xytext=(8, 6), fontsize=9)
nf = sorted([c for c in CFG if c.startswith("yolov8n")], key=lambda c: float(master[c]["npu_inf_s"]))
plt.plot([float(master[c]["npu_inf_s"]) for c in nf], [float(master[c]["test_mAP50"]) for c in nf],
         "--", color=C["n"], alpha=.6, label="YOLOv8n Pareto frontier")
plt.scatter([], [], c=C["s"], marker="s", label="YOLOv8s (off-frontier)")
plt.xlabel("NPU throughput (inf/s, RV1126 INT8)"); plt.ylabel("test mAP@0.5")
plt.title("Accuracy vs. NPU speed"); plt.grid(alpha=.3); plt.legend(loc="lower left"); plt.tight_layout()
plt.savefig(OUT / "fig2_pareto_accuracy_vs_speed.png", dpi=150); plt.close()

# 3) FP32 vs INT8 mAP (quantization gap)
plt.figure(figsize=(6.4, 4.2))
x = np.arange(len(CFG)); w = 0.38
fp = [float(acc[c]["fp32_mAP50"]) for c in CFG]; i8 = [float(acc[c]["int8_mAP50"]) for c in CFG]
plt.bar(x - w/2, fp, w, label="FP32 (host)", color=C["fp32"])
plt.bar(x + w/2, i8, w, label="INT8 (on-device NPU)", color=C["int8"])
for j in range(len(CFG)):
    plt.text(x[j] + w/2, i8[j] + .008, f"{i8[j]-fp[j]:+.3f}", ha="center", fontsize=8)
plt.xticks(x, LAB); plt.ylabel("mAP@0.5"); plt.ylim(0, 0.85)
plt.title("Quantization: FP32 vs. on-device INT8 (Δ shown)"); plt.legend(); plt.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig(OUT / "fig3_fp32_vs_int8.png", dpi=150); plt.close()

# 4) End-to-end fps: Python vs C++ vs fused
byvar = {(r["config"], r["variant"]): r for r in opt}
plt.figure(figsize=(7, 4.3))
x = np.arange(len(CFG)); w = 0.26
for k, var, col in [(0, "python", C["py"]), (1, "cpp", C["cpp"]), (2, "cpp_fused", C["fused"])]:
    vals = [float(byvar[(c, var)]["e2e_fps"]) for c in CFG]
    b = plt.bar(x + (k-1)*w, vals, w, label={"python": "Python", "cpp": "C++/NEON", "cpp_fused": "C++ + NPU-fused"}[var], color=col)
    for j in range(len(CFG)): plt.text(x[j]+(k-1)*w, vals[j]+.2, f"{vals[j]:.1f}", ha="center", fontsize=7.5)
plt.xticks(x, LAB); plt.ylabel("End-to-end throughput (fps)")
plt.title("End-to-end system fps by post-processing implementation"); plt.legend(); plt.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig(OUT / "fig4_e2e_fps.png", dpi=150); plt.close()

# 5) Stage breakdown: Python vs C++ (stacked)
plt.figure(figsize=(7.2, 4.3))
x = np.arange(len(CFG)); w = 0.36
def stack(var, xoff):
    pre = np.array([float(byvar[(c, var)]["preprocess_ms"]) for c in CFG])
    inf = np.array([float(byvar[(c, var)]["infer_ms"]) for c in CFG])
    post = np.array([float(byvar[(c, var)]["postprocess_ms"]) for c in CFG])
    plt.bar(x+xoff, pre, w, color=C["pre"]); plt.bar(x+xoff, inf, w, bottom=pre, color=C["inf"])
    plt.bar(x+xoff, post, w, bottom=pre+inf, color=C["post"])
    for j in range(len(CFG)):
        tot = pre[j]+inf[j]+post[j]
        plt.text(x[j]+xoff, tot+4, f"{tot:.0f}", ha="center", fontsize=7.5)
stack("python", -w/2); stack("cpp", w/2)
plt.xticks(x, LAB); plt.ylabel("latency (ms/frame)  —  left=Python, right=C++")
from matplotlib.patches import Patch
plt.legend(handles=[Patch(color=C["pre"], label="preprocess"), Patch(color=C["inf"], label="NPU infer"),
                    Patch(color=C["post"], label="postprocess")], loc="upper left")
plt.title("Per-stage latency: Python (left bar) vs C++/NEON (right bar)")
plt.grid(axis="y", alpha=.3); plt.tight_layout(); plt.savefig(OUT / "fig5_stage_breakdown.png", dpi=150); plt.close()

# 6) Counting error (head + total MAE, INT8)
head = {r["config"]: r for r in cnt if r["target"] == "head"}
total = {r["config"]: r for r in cnt if r["target"] == "total"}
plt.figure(figsize=(6.6, 4.2))
x = np.arange(len(CFG)); w = 0.38
hm = [float(head[c]["int8_MAE"]) for c in CFG]; tm = [float(total[c]["int8_MAE"]) for c in CFG]
plt.bar(x - w/2, hm, w, label="head-count MAE", color="#00897b")
plt.bar(x + w/2, tm, w, label="total-count MAE", color="#5e35b1")
for j in range(len(CFG)):
    plt.text(x[j]-w/2, hm[j]+.1, f"{hm[j]:.1f}", ha="center", fontsize=8)
    plt.text(x[j]+w/2, tm[j]+.1, f"{tm[j]:.1f}", ha="center", fontsize=8)
plt.xticks(x, LAB); plt.ylabel("MAE (counts/image, on-device INT8)")
plt.title("Counting error (lower = better; scenes avg 14.5 heads, 33.3 total)")
plt.legend(); plt.grid(axis="y", alpha=.3); plt.tight_layout()
plt.savefig(OUT / "fig6_counting_mae.png", dpi=150); plt.close()

print("wrote figures to", OUT)
for p in sorted(OUT.glob("*.png")): print(" ", p.name)
