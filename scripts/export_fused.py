# NPU-fused: apply DFL (softmax over 16 bins + weighted sum) INSIDE the graph,
# so the head outputs 4 distances per anchor instead of 64 logits.
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import Detect
from pathlib import Path

def fused_forward(self, x):
    out = []
    proj = torch.arange(16, dtype=torch.float32)
    for i in range(self.nl):
        box = self.cv2[i](x[i])              # [B,64,H,W]
        B, _, H, W = box.shape
        box = box.view(B, 4, 16, H, W).softmax(2)
        dist = (box * proj.view(1, 1, 16, 1, 1)).sum(2)   # [B,4,H,W] DFL in-graph
        out.append(dist)
        out.append(self.cv3[i](x[i]))        # cls logits
    return out

Detect.forward = fused_forward
m = YOLO("runs/yolov8n_320/weights/best.pt")
f = m.export(format="onnx", imgsz=320, opset=12, simplify=False, dynamic=False)
Path("onnx_rk").mkdir(exist_ok=True)
Path(f).replace("onnx_rk/yolov8n_320_fused.onnx")
print("exported fused ONNX")
