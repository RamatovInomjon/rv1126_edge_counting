import sys
from rknn.api import RKNN
# usage: convert_rknn.py <onnx> <out_rknn> <calib_txt> [platform]
onnx, out, calib = sys.argv[1], sys.argv[2], sys.argv[3]
platform = sys.argv[4] if len(sys.argv) > 4 else "rv1126"
rknn = RKNN(verbose=False)
# yolov8 preprocessing: x/255 -> mean 0, std 255
rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=platform)
assert rknn.load_onnx(model=onnx) == 0, "load_onnx failed"
assert rknn.build(do_quantization=True, dataset=calib) == 0, "build failed"
assert rknn.export_rknn(out) == 0, "export failed"
print("RKNN_OK", out)
rknn.release()
