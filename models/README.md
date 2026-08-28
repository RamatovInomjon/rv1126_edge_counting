# Models

## `pytorch_fp32/`

FP32 Ultralytics YOLOv8 checkpoints, one per configuration. These are the models
evaluated in the accuracy table and the source for every export below.

| File | Backbone | Input | Params |
|---|---|---:|---:|
| `yolov8n_320.pt` | YOLOv8n | 320 | 3.01 M |
| `yolov8n_448.pt` | YOLOv8n | 448 | 3.01 M |
| `yolov8n_640.pt` | YOLOv8n | 640 | 3.01 M |
| `yolov8s_320.pt` | YOLOv8s | 320 | 11.14 M |

Two classes: `0 = head`, `1 = person`.

## `rknn_int8/`

INT8 models converted for the Rockchip RV1126 with `rknn-toolkit` 1.7.5,
asymmetric UINT8 quantization, `mean=0 / std=255`, 20-image calibration set.
These run on the NPU; the accuracy they deliver is in
`../data/accuracy_fp32_vs_int8.csv`.

Files ending `_fused` additionally carry the DFL softmax inside the graph. They
convert and run correctly, but the fusion is **not recommended**: it saves about
0.5 ms of CPU time and costs NPU time, and is net-negative at 640 px (see
`../data/optimized_pipeline.csv`).

These `.rknn` files target RKNPU **v1** only. They will not load on RK3562 /
RK3566 / RK3588, which need `rknn-toolkit2`.

## License

Derived from Ultralytics YOLOv8 (AGPL-3.0). Use of these weights is subject to
that license.
