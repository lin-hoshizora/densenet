def gen_tfr(ds: list, folder: str, save_path: str, resize_h: int = None) -> None:


def split(path: str, n_val: int) -> None:
  """
  Split a tfrecord file to a train set and a validation set
  """
  ds = tf.data.TFRecordDataset(path)
  with tf.io.TFRecordWriter(path.replace(".tfrecord", "_train.tfrecord")) as f_train:
    with tf.io.TFRecordWriter(path.replace(".tfrecord", "_val.tfrecord")) as f_val:
      for idx, data in enumerate(ds):
        if idx < n_val:
          f_val.write(data.numpy())
        else:
          f_train.write(data.numpy())
        print(f"\r{idx + 1} done", end="")
  print()
