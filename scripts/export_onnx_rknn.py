# Export yolov8 to rknn-optimized ONNX (airockchip-style): the Detect head
# returns raw box(64ch)+cls(nc) per scale, no DFL/decode -> rknpu1-friendly.
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import Detect
from pathlib import Path

def rknn_forward(self, x):
    out = []
    for i in range(self.nl):
        out.append(self.cv2[i](x[i]))   # box distribution, 4*reg_max=64
        out.append(self.cv3[i](x[i]))   # class logits, nc
    return out

Detect.forward = rknn_forward

CFG = [("yolov8n_320",320),("yolov8n_448",448),("yolov8n_640",640),("yolov8s_320",320)]
out = Path("onnx_rk"); out.mkdir(exist_ok=True)
for name, sz in CFG:
    pt = f"runs/{name}/weights/best.pt"
    m = YOLO(pt)
    f = m.export(format="onnx", imgsz=sz, opset=12, simplify=False, dynamic=False)
    Path(f).replace(out/f"{name}.onnx")
    print(f"OK {name} imgsz={sz}")
