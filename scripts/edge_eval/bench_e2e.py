#!/usr/bin/env python3
# End-to-end per-frame timing on the device: preprocess + NPU infer + decode/NMS.
# usage: python bench_e2e.py <rknn> <imgsz> <images_dir> [N]
import time, glob, sys, numpy as np, cv2
import edge_infer as E
model, sz, images = sys.argv[1], int(sys.argv[2]), sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 40
run = E.make_runner("rknn", model, sz)
imgs = sorted(glob.glob(images + "/*.jpg"))[:N]
# preload (isolate compute from disk; live capture replaces imread)
frames = [cv2.imread(p)[:, :, ::-1] for p in imgs]
for f in frames[:3]:  # warmup
    lb, s, px, py = E.letterbox(f, sz); run(np.ascontiguousarray(lb))
tp = ti = to = 0.0
for f in frames:
    H, W = f.shape[:2]
    a = time.time(); lb, s, px, py = E.letterbox(f, sz); lb = np.ascontiguousarray(lb); tp += time.time()-a
    a = time.time(); outs = run(lb); ti += time.time()-a
    a = time.time(); E.postprocess(outs, sz, s, px, py, W, H); to += time.time()-a
n = len(frames)
pre, inf, post = tp/n*1000, ti/n*1000, to/n*1000
e2e = pre + inf + post
print("E2E %-16s pre=%.1f inf=%.1f post=%.1f => %.1f ms  %.1f fps" % (
    model.split('/')[-1], pre, inf, post, e2e, 1000.0/e2e))
