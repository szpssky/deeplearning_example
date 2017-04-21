import tensorflow as tf
import os
from scipy import ndimage
import sys

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_png_data = tf.placeholder(dtype=tf.string)
    self._decode_png = tf.image.decode_png(self._decode_png_data, channels=1)

  def read_image_dims(self, sess, image_data):
    image = self.decode_png(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_png(self, sess, image_data):
    image = sess.run(self._decode_png,
                     feed_dict={self._decode_png_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 1
    return image


def int64_feature(values):
    """Returns a TF-Feature of int64s.
    Args:
      values: A scalar or list of values.
    Returns:
      a TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    """Returns a TF-Feature of bytes.
    Args:
      values: A string.
    Returns:
      a TF-Feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(image_data, image_format, height, width, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(image_format),
        'image/class/label': int64_feature(class_id),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
    }))


def add_to_tfrecord(class_path, tfrecord_writer,max_size):
    with tf.Graph().as_default():
        image_reader = ImageReader()
        for label, name in enumerate(classes):
            path = os.path.join(class_path, name)
            image_files = os.listdir(path)
            count = 0
            with tf.Session() as sess:
                for image_name in image_files:
                    try:
                        image_file = os.path.join(path, image_name)
                        # image = ndimage.imread(image_file)
                        image = tf.gfile.FastGFile(image_file, 'r').read()
                        height, width = image_reader.read_image_dims(sess, image)

                        if height!=28 | width != (28, 28):
                            raise Exception('Unexpected image shape: %s' % str(image.shape))
                        # image = tf.reshape(image, [28, 28, 1])
                        # encode_png = tf.image.encode_png(image)

                        sys.stdout.write('\r>> Converting Class %s image %d/%d' % (name, count + 1, image_files.__len__()))
                        sys.stdout.flush()
                        # png_string = sess.run(image)

                        example = image_to_tfexample(image, 'png'.encode(), 28, 28, label)
                        tfrecord_writer.write(example.SerializeToString())
                        count += 1
                    except Exception as e:
                        print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
                    if count == max_size:
                        break;


with tf.python_io.TFRecordWriter('test231.tfrecord') as tfrecord_writer:
    add_to_tfrecord('../../notMNIST_small/', tfrecord_writer,1000)
#
# with tf.python_io.TFRecordWriter('train.tfrecord') as tfrecord_writer:
#     add_to_tfrecord('../../notMNIST_large/', tfrecord_writer,20000)
