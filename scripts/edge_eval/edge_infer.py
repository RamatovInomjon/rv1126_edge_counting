#!/usr/bin/env python3
# Backend-agnostic YOLOv8 eval: decode (DFL, anchor-free) + NMS, then either
# compute mAP (host) or dump predictions (device). Same decode for ONNX(FP32)
# and RKNN(INT8) so FP32 vs INT8 is a fair comparison.
# Python 3.7 compatible (runs on the RV1126 too).
import os, sys, json, argparse, glob
import numpy as np
import cv2

NC = 2
NAMES = ["head", "person"]
REG = 16

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def softmax(x, axis):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)

def letterbox(im, sz):
    h, w = im.shape[:2]
    s = min(sz / h, sz / w)
    nh, nw = int(round(h * s)), int(round(w * s))
    r = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((sz, sz, 3), 114, np.uint8)
    py, px = (sz - nh) // 2, (sz - nw) // 2
    out[py:py+nh, px:px+nw] = r
    return out, s, px, py

def decode(outs, sz, conf=0.001):
    # group 6 outputs into 3 (box[64], cls[NC]) scale pairs
    pairs = []
    for k in range(0, len(outs), 2):
        a, b = outs[k], outs[k+1]
        if a.shape[1] != 4 * REG: a, b = b, a   # a=box(64ch), b=cls
        pairs.append((a, b))
    B, S, C = [], [], []
    for box, cls in pairs:
        _, _, H, W = box.shape
        stride = sz // H
        gx, gy = np.meshgrid(np.arange(W), np.arange(H))
        ax = (gx + 0.5).reshape(-1); ay = (gy + 0.5).reshape(-1)
        bd = box.reshape(4, REG, H, W)
        dist = (softmax(bd, 1) * np.arange(REG).reshape(1, REG, 1, 1)).sum(1).reshape(4, -1)
        l, t, r, b = dist
        x1 = (ax - l) * stride; y1 = (ay - t) * stride
        x2 = (ax + r) * stride; y2 = (ay + b) * stride
        sc = sigmoid(cls.reshape(NC, -1))
        score = sc.max(0); cid = sc.argmax(0)
        m = score > conf
        B.append(np.stack([x1, y1, x2, y2], 1)[m]); S.append(score[m]); C.append(cid[m])
    return np.concatenate(B), np.concatenate(S), np.concatenate(C)

def nms(boxes, scores, iou_thr):
    x1, y1, x2, y2 = boxes.T
    area = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1); h = np.maximum(0, yy2 - yy1)
        inter = w * h
        ov = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][ov <= iou_thr]
    return keep

def postprocess(outs, sz, s, px, py, W, H, iou_thr=0.7, max_det=300):
    boxes, scores, cls = decode(outs, sz)
    if len(boxes) == 0: return np.zeros((0, 6))
    # letterbox -> original pixels -> normalized [0,1]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - px) / s
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - py) / s
    out = []
    for c in range(NC):
        m = cls == c
        if not m.any(): continue
        b, sc = boxes[m], scores[m]
        k = nms(b, sc, iou_thr)
        for j in k:
            out.append([b[j, 0]/W, b[j, 1]/H, b[j, 2]/W, b[j, 3]/H, sc[j], c])
    out = np.array(out) if out else np.zeros((0, 6))
    if len(out) > max_det:
        out = out[out[:, 4].argsort()[::-1][:max_det]]
    return out

# ---------------- backends ----------------
def make_runner(backend, model, sz):
    if backend == "onnx":
        import onnxruntime as ort
        sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0].name
        def run(x):  # x: HWC uint8 letterboxed
            xf = x.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
            return sess.run(None, {inp: xf})
        return run
    else:
        from rknnlite.api import RKNNLite
        r = RKNNLite(); assert r.load_rknn(model) == 0; assert r.init_runtime() == 0
        def run(x):  # rknn does /255 internally; feed uint8 HWC
            return r.inference(inputs=[x])
        return run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["onnx", "rknn"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run = make_runner(a.backend, a.model, a.imgsz)
    imgs = sorted(glob.glob(os.path.join(a.images, "*.jpg")))
    preds = {}
    for n, p in enumerate(imgs):
        im = cv2.imread(p)
        if im is None: continue
        im = im[:, :, ::-1]  # BGR->RGB
        H, W = im.shape[:2]
        lb, s, px, py = letterbox(im, a.imgsz)
        outs = run(np.ascontiguousarray(lb))
        det = postprocess(outs, a.imgsz, s, px, py, W, H)
        preds[os.path.splitext(os.path.basename(p))[0]] = det.tolist()
        if (n + 1) % 50 == 0: print("  %d/%d" % (n + 1, len(imgs)))
    json.dump(preds, open(a.out, "w"))
    print("wrote %s (%d images)" % (a.out, len(preds)))

if __name__ == "__main__":
    main()
