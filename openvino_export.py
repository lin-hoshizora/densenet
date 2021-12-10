import sys
from openvino.inference_engine import IENetwork, IECore

assert len(sys.argv) == 3, "Usage: python openvino_export.py <MODEL_PATH> <DEV>"
model_path = sys.argv[1]
dev = sys.argv[2]


ie_core = IECore()
ie_core.set_config({"VPU_MYRIAD_FORCE_RESET": "YES"}, "MYRIAD")
net = ie_core.read_network(model=model_path+".xml", weights=model_path+".bin")
exe = ie_core.load_network(net, dev, num_requests=1)
exe.export(model_path + "_" + dev.upper())
