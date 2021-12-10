from ..utils import load_conf, ensure_exist

class LabelParser:
  def get_ds_from(self, conf_path):
    conf = load_conf(conf_path)
    ensure_exist()

np.random.shuffle(raw_labels)
n_val_raw = int(len(raw_labels)*0.1)
raw_labels_val = raw_labels[:n_val_raw]
raw_labels_train = raw_labels[n_val_raw:]
labels_val = pad(labels_val)
labels_train = pad(labels_train)
x_val = [x[0] for x in labels_val]
y_val = np.array([x[1] for x in labels_val])


@tf.function
def parse_train(p, lb):
    aug_dice = tf.random.uniform(())
    if aug_dice < 0.2:
        p = tf.strings.regex_replace(p, '/combined13/', '/combined13_blur7/')
    elif aug_dice < 0.4:
        p = tf.strings.regex_replace(p, '/combined13/', '/combined13_blur5/')
    elif aug_dice < 0.6:
        p = tf.strings.regex_replace(p, '/combined13/', '/combined13_elastic/')
    elif aug_dice < 0.8:
        p = tf.strings.regex_replace(p, '/combined13/', '/combined13_b53d2/')
    img = tf.io.decode_jpeg(tf.io.read_file(p))
    img_shape = tf.cast(tf.shape(img), tf.float32)
    if tf.strings.regex_full_match(p, '.+combine.+'):
        crop_max = 0.15
    elif tf.strings.regex_full_match(p, '.+mincho.+'):
        crop_max = 0.05
    else:
        crop_max = 0.1
    crop_ratio = tf.random.uniform(()) * crop_max
    crop_h = img_shape[0] / (1 + crop_ratio)
    crop_w = tf.cast(img_shape[1], tf.int32) - tf.cast(crop_h * crop_ratio, tf.int32)
    crop_h = tf.cast(crop_h, tf.int32)
    img = tf.image.random_crop(img, (crop_h, crop_w, 3))
    img_shape = tf.cast(tf.shape(img), tf.float32)
    w = tf.cast(HEIGHT / img_shape[0] * img_shape[1], tf.int64)
    img = tf.image.resize(img, (int(HEIGHT), w), method=tf.image.ResizeMethod.AREA)
    lb = tf.cast(lb, tf.int64)
    lb_len = tf.reduce_sum(tf.cast(tf.not_equal(lb, -1), tf.int64))
    input_len = w // int(HEIGHT / 4)
    return img, lb, lb_len, input_len

@tf.function
def parse_val(p, lb):
    img = tf.io.decode_jpeg(tf.io.read_file(p))
    img_shape = tf.cast(tf.shape(img), tf.float32)
    if tf.strings.regex_full_match(p, '.+combine.+'):
        crop_max = 0.075
        crop_ratio = tf.random.uniform(()) * crop_max
        crop_h = img_shape[0] / (1 + crop_ratio)
        crop_w = tf.cast(img_shape[1], tf.int32) - tf.cast(crop_h * crop_ratio, tf.int32)
        crop_h = tf.cast(crop_h, tf.int32)
        img = tf.image.random_crop(img, (crop_h, crop_w, 3))
        img_shape = tf.cast(tf.shape(img), tf.float32)
    w = tf.cast(HEIGHT / img_shape[0] * img_shape[1], tf.int64)
    img = tf.image.resize(img, (int(HEIGHT), w), method=tf.image.ResizeMethod.AREA)
    lb = tf.cast(lb, tf.int64)
    lb_len = tf.reduce_sum(tf.cast(tf.not_equal(lb, -1), tf.int64))
    input_len = w // int(HEIGHT / 4)
    return img, lb, lb_len, input_len

@tf.function
def augment(img, lb, lb_len, input_len, delta=0.15):
    img = tf.cast(img, tf.uint8)
    img = tf.image.random_jpeg_quality(img, min_jpeg_quality=50, max_jpeg_quality=95)
#     img = tf.image.random_brightness(img, delta)
    img = tf.image.random_contrast(img, 1 - delta, 1 + delta)
    img = tf.image.random_hue(img, delta)
    img = tf.image.random_saturation(img, 1 - delta, 1 + delta)
    img = tf.cast(img, tf.float32)
    
    if SP_NOISE:
        if tf.random.uniform(()) > 0.5:
            dice = tf.random.uniform(tf.shape(img)[:-1])
            dice = tf.expand_dims(dice, axis=-1)
            dice = tf.concat([dice]*3, axis=-1)
            white = tf.fill(tf.shape(img), tf.random.uniform(shape=[], minval=128., maxval=255.))
            black = tf.fill(tf.shape(img), tf.random.uniform(shape=[], minval=0., maxval=128.))
            img = tf.where(tf.less(dice, 0.01), white, img)
            img = tf.where(tf.greater(dice, 1-0.01), black, img)
    return img, lb, lb_len, input_len

@tf.function
def norm(img, lb, lb_len, input_len):
    img = tf.cast(img, tf.float32)
    return {'image': img, 'labels': lb, 'label_lengths': lb_len, 'input_lengths': input_len}
x_train = [x[0] for x in labels_train]
y_train = np.array([x[1] for x in labels_train])

AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 512
# BUF_SIZE =  int(10712 / (BATCH_SIZE / 256))
BUF_SIZE = len(x_train)
SP_NOISE = True

ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(BUF_SIZE).map(parse_train, num_parallel_calls=AUTO)
ds_train = ds_train.map(augment, num_parallel_calls=AUTO)
ds_train = ds_train.padded_batch(BATCH_SIZE).map(norm, num_parallel_calls=AUTO).prefetch(AUTO)

ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).map(parse_val, num_parallel_calls=AUTO)
ds_val = ds_val.padded_batch(BATCH_SIZE).map(norm, num_parallel_calls=AUTO).prefetch(AUTO)
