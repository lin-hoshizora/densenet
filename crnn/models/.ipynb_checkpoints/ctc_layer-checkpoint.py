import tensorflow as tf
from tensorflow import keras


class CTCLayer(keras.layers.Layer):
  """
  A dumb layer to calculate CTC loss and edit distance
  """
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
