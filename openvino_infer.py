import sys
import time
from pathlib import Path
import pickle
import numpy as np
import cv2
from openvino.inference_engine import IENetwork, IECore


def resize_pad(img, h, w):
  w_new = int(h / img.shape[0] * img.shape[1])
  img = cv2.resize(img, (w_new, h), cv2.INTER_AREA)
  if img.shape[1] > w: return img[:, :w], w // 16
  padded = np.zeros((h, w, 3)).astype(img.dtype)
  padded[:, :img.shape[1]] = img
  return padded, img.shape[1] // 16


def greedy_decode(x: np.ndarray, length: int):
  """
  CTC greedy decoder
  :param x: CTC encoded sequence, last label as void
  :param length: sequence length
  :return: decoded sequence and probability for each char
  """
  lb_void = x.shape[1] - 1
  encodes = x.argmax(axis=1)
  probs = [x[r][i] for r, i in enumerate(encodes)]
  decodes = []
  dec_prob = []
  positions = []
  prev = -1
  for i, code in enumerate(encodes[:length]):
    if code != lb_void:
      if prev == lb_void or code != prev:
        decodes.append(code)
        dec_prob.append(probs[i])
        positions.append(i)
      else:
        if probs[i] > dec_prob[-1]:
          dec_prob[-1] = probs[i]
    prev = code
  decodes = np.array(decodes)
  dec_prob = np.array(dec_prob)
  positions = np.array(positions)
  return decodes, dec_prob, positions

if __name__ == "__main__":
  assert len(sys.argv) == 4, "Usage: python openvino_infer <MODEL PATH> <IMAGE PATH> <DEVICE>"
  model_path = sys.argv[1]
  img_path = sys.argv[2]
  dev = sys.argv[3]
  n_iter = 10


  ie_core = IECore()
  # assume an exported model is used when xml and bin files cannot be found
  exported = not (Path(model_path + ".xml").exists() and Path(model_path + ".bin").exists())
  if exported:
    exe = ie_core.import_network(model_path, dev, num_requests=1)
  else:
    net = ie_core.read_network(model=model_path+".xml", weights=model_path+".bin")
    exe = ie_core.load_network(net, dev, num_requests=1)

  # get IO info
  input_name = next(iter(exe.input_info))
  input_shape = exe.input_info[input_name].input_data.shape
  print("input shape", input_shape)
  output_name = next(iter(exe.outputs))

  # prepare input data
  img = cv2.imread(img_path)[..., ::-1]
  img, input_len = resize_pad(img, 64, input_shape[3])
  input_data = img.transpose([2, 0, 1])[np.newaxis, ...].astype(np.float32)

  # run inference
  openvino_times = []
  for _ in range(n_iter):
    t0 = time.time()
    req = exe.start_async(exe.get_idle_request_id(), {input_name: input_data})
    req.wait()
    openvino_out = req.output_blobs[output_name].buffer
    t = time.time() - t0
    openvino_times.append(t)
    print(f"OpenVINO inference: {t:.3f}s")

  prob = openvino_out[0]
  codes, probs, positions = greedy_decode(prob, input_len)

  with open("id2char_std.pkl", "rb") as f:
    id2char = pickle.load(f)

  text = "".join([id2char[x] for x in codes])
  print("Predicted text:", text)
  for c, p in zip(text, probs):
    print(c, f"{p * 100 :.2f}")
