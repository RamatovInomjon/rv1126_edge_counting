# Runs ON the camera (rknnlite). Times pure NPU inference for a given .rknn.
# usage: python bench_device.py <rknn_path> <imgsz> [iters]
import sys, time, numpy as np
from rknnlite.api import RKNNLite
path = sys.argv[1]; sz = int(sys.argv[2]); N = int(sys.argv[3]) if len(sys.argv) > 3 else 50
r = RKNNLite()
assert r.load_rknn(path) == 0, "load failed"
assert r.init_runtime() == 0, "init failed"
x = np.random.randint(0, 255, (sz, sz, 3), dtype=np.uint8)
for _ in range(5): r.inference(inputs=[x])       # warmup
t = time.time()
for _ in range(N): r.inference(inputs=[x])
ms = (time.time() - t) / N * 1000
print("BENCH %s  in=%dx%d  %.1f ms  %.1f inf/s" % (path.split('/')[-1], sz, sz, ms, 1000.0/ms))
r.release()
