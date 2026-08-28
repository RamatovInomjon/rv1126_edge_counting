# Scripts

Run in this order. Set the dataset location first:

```bash
export CROWDHUMAN_SRC=/path/to/crowdhuman/all        # images/<split>, labels/<split>
export CROWDHUMAN_SUBSET=/path/to/crowdhuman_subset  # created by make_subset.py
```

| Step | Script | Runs on | Purpose |
|---|---|---|---|
| 1 | `make_subset.py` | host | Deterministic 3000/600/300 subset |
| 2 | `train_matrix.py` | host GPU | Trains the four configurations, evaluates on test |
| 3 | `export_onnx_rknn.py` | host | Decoupled-head ONNX export (RKNPU v1 friendly) |
| 4 | `convert_rknn.py` | host | INT8 quantization for `rv1126`, uses `calib20.txt` |
| 5 | `bench_device.py` | **device** | Isolated NPU latency benchmark (RKNNLite) |
| 6 | `edge_eval/map_eval.py` | host + device | FP32 vs INT8 accuracy, shared decoder |
| 7 | `edge_eval/counting_eval.py` | host | Counting MAE / RMSE / bias, tunes the threshold on val |
| 8 | `edge_eval/bench_e2e.py` | **device** | Reference Python pipeline timing |
| 9 | `edge_eval/cpp/bench_e2e.cpp` | **device** | C++/NEON pipeline (`--fused` for NPU-side DFL) |
| 10 | `export_fused.py` | host | Builds the NPU-fused DFL ONNX variant |
| 11 | `make_figures.py` | host | Regenerates `../figures/fig1`–`fig6` |

`edge_eval/edge_infer.py` is the shared NumPy YOLOv8 decoder (DFL expectation,
per-class sigmoid, per-class NMS) used by steps 6–8 so that FP32 and INT8 are
scored by identical code.

Building the C++ benchmark on the device:

```bash
g++ -O3 -mfpu=neon -o bench_e2e edge_eval/cpp/bench_e2e.cpp -lrknn_api
```

Steps marked **device** must run on the RV1126 itself; the rest run on the
training host.
