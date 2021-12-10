import tensorflow as tf
from tensorflow import keras


class CTCLayer(keras.layers.Layer):
    
    def call(self, labels, logits, input_length, label_length):
        input_length = tf.cast(input_length, tf.int32)
        label_length = tf.cast(label_length, tf.int32)
        loss = keras.backend.ctc_batch_cost(
            labels, tf.nn.softmax(logits, axis=-1), input_length[:, tf.newaxis], label_length[:, tf.newaxis]
        )
        self.add_loss(tf.reduce_mean(loss))
    
        decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(tf.transpose(logits, [1, 0, 2]), sequence_length=input_length)
        label_sparse = keras.backend.ctc_label_dense_to_sparse(labels, label_length)
        edit_distance = tf.edit_distance(decoded[0], label_sparse)
        self.add_metric(tf.reduce_mean(edit_distance), name='edit_distance', aggregation='mean')
        return logits

HEIGHT = 64.
config = {
    "alpha": 1.0,
    "minimalistic": True,
    "weights": None,
    "include_top": False
}

def get_model(infer=True, input_shape=(None, None, 3)):
  image = keras.Input(shape=input_shape, name="image")
  base = keras.applications.MobileNetV3Large(input_tensor=image, **config)
  if HEIGHT == 64:
      feature = base.get_layer("expanded_conv_11/Add").output
  if HEIGHT == 32:
      feature = base.get_layer("expanded_conv_5/Add").output
  logits = keras.layers.SeparableConv2D(7550, kernel_size=(4, 1),
                                       strides=1, padding="valid",
                                       depthwise_initializer="he_normal")(feature)
  logits = keras.layers.Lambda(lambda x: tf.squeeze(x, 1), name='logits')(logits)
  model = None
  if infer:
    probs = keras.layers.Softmax(name="probs")(logits)
    model = keras.Model(inputs=image, outputs=probs)
  else:
    labels = keras.Input(shape=(None,), dtype=tf.int64, name='labels')
    label_lengths = keras.Input(shape=(), dtype=tf.int64, name='label_lengths')
    input_lengths = keras.Input(shape=(), dtype=tf.int64, name='input_lengths')
    logits = CTCLayer()(labels, logits, input_lengths, label_lengths)
    model = keras.Model(inputs=[base.input, labels, input_lengths, label_lengths], outputs=logits)

  return model
