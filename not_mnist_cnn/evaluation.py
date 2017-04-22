import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
from preprocessing import inception_preprocessing
import notMNIST

image_size = nets.inception.inception_v1.default_image_size
train_log_dir = './variables/'


def load_batch(dataset, batch_size=32, height=299, width=299, is_training=False):
    """Loads a single batch of data.

    Args:
      dataset: The dataset to load.
      batch_size: The number of images in the batch.
      height: The size of each image after preprocessing.
      width: The size of each image after preprocessing.
      is_training: Whether or not we're currently training or evaluating.

    Returns:
      images: A Tensor of size [batch_size, height, width, 3], image samples that have been preprocessed.
      images_raw: A Tensor of size [batch_size, height, width, 3], image samples that can be used for visualization.
      labels: A Tensor of size [batch_size], whose values range between 0 and dataset.num_classes.
    """
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, common_queue_capacity=32,
        common_queue_min=8)
    image_raw, label = data_provider.get(['image', 'label'])

    # Preprocess image for usage by Inception.
    image = inception_preprocessing.preprocess_image(image_raw, height, width, is_training=is_training)
    # Preprocess the image for display purposes.
    image_raw = tf.expand_dims(image_raw, 0)
    image_raw = tf.image.resize_images(image_raw, [height, width])
    image_raw = tf.squeeze(image_raw)

    # Batch it up.
    images, images_raw, labels = tf.train.batch(
        [image, image_raw, label],
        batch_size=batch_size,
        num_threads=1,
        capacity=200000)

    return images, images_raw, labels


slim = tf.contrib.slim

with tf.Graph().as_default():
    test_dataset = notMNIST.get_split('test', './')
    test_image, _, test_labels = load_batch(test_dataset, height=image_size, width=image_size)
    test_logits, _ = nets.inception.inception_v1(test_image, test_dataset.num_classes, is_training=False)
    test_predict = tf.argmax(test_logits, 1)

    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'eval/Accuracy': slim.metrics.streaming_accuracy(test_predict, test_labels),
    })

    print('Running evaluation Loop...')
    checkpoint_path = tf.train.latest_checkpoint(train_log_dir)

    initial_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer())

    with tf.Session() as sess:
        metric_values = slim.evaluation(
            sess,
            num_evals=1,
            inital_op=initial_op,
            eval_op=names_to_updates.values(),
            final_op=names_to_values.values())

        names_to_values = dict(zip(names_to_values.keys(), metric_values))
        for name in names_to_values:
            print('%s: %f' % (name, names_to_values[name]))
