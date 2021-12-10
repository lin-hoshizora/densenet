"""
CRNN-like model
"""
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from . import backbone
from .ctc_layer import CTCLayer
from ..utils import load_conf, conf_folder


def get_backbone(conf_path, input_tensor):
  """
  Create backbone network based on given config
  """
  conf = load_conf(str(conf_folder / (conf_path + ".yaml")))
  model = getattr(backbone, conf["name"])(**conf["options"], input_tensor=input_tensor)
  return model


def crnn(conf, train=True, infer_shape=None, include_softmax=True):
  """
  Create a CRNN-like network for text recognition
  """
  if not train:
    if infer_shape is None:
      raise ValueError("You haved to specify input shape for inference")
    conf["input_shape"] = infer_shape

  image = keras.Input(shape=conf["input_shape"], name="image")
  base = get_backbone(conf["backbone"], input_tensor=image)
  feature = base.get_layer(conf["feature_layer"]).output
  logits_ori = keras.layers.SeparableConv2D(conf["n_class"], kernel_size=(4, 1),
                                       strides=1, padding="valid",
                                       depthwise_initializer=conf["kernel_init"])(feature)
  logits = keras.layers.Lambda(lambda x: tf.squeeze(x, 1), name='logits')(logits_ori)

  if train:
    labels = keras.Input(shape=(None,), dtype=tf.int64, name='labels')
    label_lengths = keras.Input(shape=(), dtype=tf.int64, name='label_lengths')
    input_lengths = keras.Input(shape=(), dtype=tf.int64, name='input_lengths')
    logits = CTCLayer()(labels, logits, input_lengths, label_lengths)
    model = keras.Model(inputs=[base.input, labels, input_lengths, label_lengths], outputs=logits)
    return model

  if not include_softmax:
    model = keras.Model(inputs=image, outputs=logits_ori)
    return model

  probs = keras.layers.Softmax(name="probs")(logits)
  model = keras.Model(inputs=image, outputs=probs)
  return model
