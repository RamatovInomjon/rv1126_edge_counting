# Experimental Results — Head/Person Counting on a Rockchip RV1126 Edge NPU

Raw results, tables, and figures for write-up. Every number is measured; the "Findings"
bullets are observations, not interpretation.

---

## 0. Setup (one paragraph)

- **Device:** Rockchip **RV1126** — quad-core Cortex-A7 (armv7l, Linux 4.19), **RKNPU v1**,
  2.0 TOPS **INT8**; on-device runtime `librknn_runtime` 1.7.0.
- **Task:** counting by detection — 2 classes `head`, `person`. Count = # detections after NMS
  and a confidence threshold.
- **Dataset:** CrowdHuman-derived subset, **3,000 train / 600 val / 300 test** (dense: avg
  14.5 heads, 18.8 persons, 33.3 objects per test image).
- **Models:** YOLOv8n @ {320, 448, 640} and YOLOv8s @ 320 (Ultralytics, 80 epochs each).
- **Pipeline:** train (PyTorch, RTX 3070) → export RKNPU-friendly ONNX → **INT8** quantize
  (`rknn-toolkit 1.7.5`, target rv1126, 20-image calibration) → deploy → benchmark on NPU.

---

## 1. Experiment A — Accuracy vs. model / resolution

Held-out **test** split (300 img). Val shown for the split-consistency check.

| Config | Params | GFLOPs | INT8 size | val mAP50 | **test mAP50** | test mAP50-95 | head AP | person AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n@320 | 3.01 M | 2.05 | 3.16 MB | 0.606 | 0.600 | 0.326 | 0.293 | 0.360 |
| YOLOv8n@448 | 3.01 M | 4.02 | 3.16 MB | 0.704 | 0.690 | 0.397 | 0.374 | 0.420 |
| **YOLOv8n@640** | 3.01 M | 8.20 | 3.16 MB | 0.790 | **0.774** | **0.464** | 0.450 | 0.479 |
| YOLOv8s@320 | 11.14 M | 7.16 | 11.30 MB | 0.655 | 0.645 | 0.368 | 0.335 | 0.400 |

**Figures:** `figures/fig1_accuracy_vs_resolution.png`, `fig2_pareto_accuracy_vs_speed.png`

**Findings**
- Accuracy rises strongly with input resolution for the same tiny model: 320→448→640 gives
  mAP50 **0.600 → 0.690 → 0.774**.
- **Resolution beats capacity at matched compute:** `n@448` (4.0 GFLOPs) > `s@320` (7.2 GFLOPs)
  by +4.5 mAP50 with 3.7× fewer params; `s@320` has ~same GFLOPs as `n@640` yet scores 12.9 pts
  lower.
- The head class (the counting-relevant one) is the most resolution-sensitive: head AP
  0.293 → 0.374 → 0.450.
- Val↔test agree within 0.6–1.6 mAP50 points; ranking identical → no overfitting.
- Pareto frontier = the three YOLOv8n points; `s@320` is off it.

---

## 2. Experiment B — INT8 quantization (FP32 vs. on-device INT8)

Same 300-image set, same NumPy decoder applied to host-FP32 (ONNX) and device-INT8 (RKNN);
decoder validated to reproduce the reference FP32 mAP within ~1 point.

| Config | FP32 mAP50 | **INT8 mAP50 (device)** | Δ | FP32 mAP50-95 | INT8 mAP50-95 | Δ |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8n@320 | 0.603 | 0.597 | −0.007 | 0.333 | 0.327 | −0.005 |
| YOLOv8n@448 | 0.687 | 0.684 | −0.003 | 0.398 | 0.393 | −0.006 |
| YOLOv8n@640 | 0.766 | 0.760 | −0.006 | 0.462 | 0.454 | −0.008 |
| YOLOv8s@320 | 0.652 | 0.647 | −0.006 | 0.379 | 0.370 | −0.009 |

**Figure:** `figures/fig3_fp32_vs_int8.png`

**Findings**
- INT8 on the RV1126 NPU is **effectively lossless**: ≤ 0.7 mAP50 drop (< 1% relative),
  ≤ 0.9 mAP50-95 drop; ranking preserved.

---

## 3. Experiment C — Counting error (MAE / RMSE / bias)

Predicted count = # post-NMS detections with score ≥ τ. **τ tuned on val** (per target) to
minimize MAE, reported on **test**. On-device INT8 (FP32 within ≈0.2 of each value).

| Config | head τ | **head MAE** | head bias | total τ | **total MAE** | total bias |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8n@320 | 0.10 | 6.10 | −3.67 | 0.175 | 9.43 | −3.33 |
| YOLOv8n@448 | 0.15 | 4.45 | −2.65 | 0.25 | 7.73 | −4.21 |
| **YOLOv8n@640** | 0.225 | **3.37** | −1.83 | 0.25 | **6.84** | −1.49 |
| YOLOv8s@320 | 0.075 | 5.79 | −3.38 | 0.175 | 8.34 | −3.36 |

**Figure:** `figures/fig6_counting_mae.png`

**Findings**
- Head-count MAE improves with resolution: 6.1 → 4.5 → 3.4; `n@640` roughly halves `n@320`.
- **All models undercount** (bias < 0) — the error is dominated by missed heads in dense crowds.
- `s@320` (head MAE 5.8) again loses to `n@448` (4.5).
- INT8 ≈ FP32 for counting too (e.g. n@640 head MAE 3.38 → 3.37).

---

## 4. Experiment D — Speed: NPU, end-to-end, and optimizations

### 4.1 NPU-only inference (INT8, RKNNLite, 50 iters)

| Config | latency | throughput |
|---|---:|---:|
| YOLOv8n@320 | 43.5 ms | 23.0 inf/s |
| YOLOv8n@448 | 75.1 ms | 13.3 inf/s |
| YOLOv8n@640 | 128.4 ms | 7.8 inf/s |
| YOLOv8s@320 | 67.6 ms | 14.8 inf/s |

### 4.2 End-to-end system throughput — three implementations (on-device, conf 0.25)

| Config | Python fps | **C++/NEON fps** | C++ +NPU-fused fps | postproc: Python → C++ → fused |
|---|---:|---:|---:|---:|
| YOLOv8n@320 | 4.7 | **26.2** | 26.6 | 140.0 → 1.1 → 0.4 ms |
| YOLOv8n@448 | 2.9 | **15.5** | 15.6 | 249.8 → 1.5 → 0.6 ms |
| YOLOv8n@640 | 2.0 | **9.1** | 8.3 | 358.5 → 2.2 → 1.1 ms |
| YOLOv8s@320 | 4.4 | **18.5** | 18.7 | 132.7 → 1.1 → 0.4 ms |

Per-stage (ms/frame), Python vs C++:

| Config | Py pre | Py inf | **Py post** | C++ pre | C++ inf | **C++ post** |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8n@320 | 44.0 | 29.9 | **140.0** | 13.2 | 23.8 | **1.1** |
| YOLOv8n@448 | 46.5 | 54.1 | **249.8** | 22.8 | 40.2 | **1.5** |
| YOLOv8n@640 | 51.1 | 95.1 | **358.5** | 44.8 | 62.5 | **2.2** |
| YOLOv8s@320 | 43.6 | 48.8 | **132.7** | 13.4 | 39.5 | **1.1** |

**Figures:** `figures/fig4_e2e_fps.png`, `fig5_stage_breakdown.png`

**Findings**
- In the reference **Python** pipeline, post-processing (NumPy DFL decode + NMS) is the
  bottleneck — **3–4× the NPU time** — capping the system at **2–5 fps**.
- A **C++/NEON** decode (class-threshold before DFL) cuts post-processing to **~1 ms**
  (**~100–160×**), lifting end-to-end **4.2–5.6×** to **9.1–26.2 fps**; inference and the
  letterbox become the new limit.
- **NPU-fused DFL** (softmax moved into the graph — it *does* convert on rknpu1) is only
  marginal (< 2%) and **net-negative at 640**: the NPU softmax over 8,400 anchors adds ≈12 ms
  to inference (62.5 → 74.3 ms), more than the ~1 ms it saves → 9.1 → 8.3 fps.
- Recommended engineering order: (1) C++/NEON decode, (2) hardware (RGA) letterbox; **not** a
  bigger model and **not** NPU-side DFL fusion.

---

## 5. Consolidated findings

1. **Resolution > model capacity** for head/person counting on this NPU (accuracy, counting
   error, and compute-efficiency all agree). Use YOLOv8n at higher resolution, not YOLOv8s.
2. **INT8 is lossless** on RV1126 (< 1% mAP), verified on-device.
3. **The decode, not the NPU, gates throughput** — until it is written in C++/NEON, which
   restores near-NPU-bound fps. NPU-side DFL fusion is not worthwhile.
4. Operating points (on-device INT8, C++ pipeline):
   - **n@320** — 26 fps, mAP50 0.60, head MAE 6.1 (speed).
   - **n@448** — 16 fps, mAP50 0.69, head MAE 4.5 (**recommended balance**).
   - **n@640** — 9 fps, mAP50 0.77, head MAE 3.4 (accuracy).

---

## 6. Figures index (`figures/`)

| File | Content |
|---|---|
| `fig1_accuracy_vs_resolution.png` | mAP vs input size (n) + s@320 marker |
| `fig2_pareto_accuracy_vs_speed.png` | accuracy vs NPU throughput, Pareto frontier |
| `fig3_fp32_vs_int8.png` | FP32 vs on-device INT8 mAP (Δ labels) |
| `fig4_e2e_fps.png` | end-to-end fps: Python / C++ / fused |
| `fig5_stage_breakdown.png` | per-stage latency, Python vs C++ (stacked) |
| `fig6_counting_mae.png` | head & total count MAE per model |

## 7. Data files

| Path | Contents |
|---|---|
| `data/results_master.csv` | Accuracy, complexity and NPU latency per config (§1) |
| `data/accuracy_fp32_vs_int8.csv` | FP32 against on-device INT8 accuracy (§2) |
| `data/counting_errors.csv` | Counting MAE / RMSE / bias per config and target (§3) |
| `data/end_to_end_fps.csv` | Reference Python pipeline, per-stage timing (§4.2) |
| `data/optimized_pipeline.csv` | Python / C++ / NPU-fused pipelines (§4.2) |
| `data/device_npu_benchmark.txt` | Raw on-device RKNNLite benchmark log (§4.1) |
| `eval_details/<config>/` | Confusion matrices, PR/P/R/F1 curves, qualitative predictions |
| `training_curves/<config>_training.png` | Training loss and validation mAP curves |
| `models/pytorch_fp32/` | FP32 `.pt` checkpoints |
| `models/rknn_int8/` | INT8 `.rknn` models for RV1126, including `_fused` variants |
