import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
import pickle
import numpy as np
import os
from preprocessing import inception_preprocessing
import notMNIST

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

image_size = nets.inception.inception_v1.default_image_size
train_log_dir = './variables/'


def get_init_fn():
    """Returns a function run by the chief worker to warm-start the training."""
    checkpoint_exclude_scopes = ["InceptionV1/Logits", "InceptionV1/AuxLogits"]

    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(
        os.path.join('./inception_v1', 'inception_v1.ckpt'),
        variables_to_restore)


with tf.Graph().as_default():

    tf.logging.set_verbosity(tf.logging.INFO)
    # dataset = flowers.get_split('train', '../flowers')
    dataset = notMNIST.get_split('train','./')
    images, _, labels = load_batch(dataset, height=image_size, width=image_size)


    one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
    with slim.arg_scope(nets.inception.inception_v1_arg_scope()):
        predictions, _ = nets.inception.inception_v1(images, dataset.num_classes, is_training=True)


    loss = slim.losses.softmax_cross_entropy(predictions, one_hot_labels)

    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('losses/total_loss', total_loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = slim.learning.create_train_op(total_loss, optimizer)

    final_loss = slim.learning.train(
        train_op,
        logdir=train_log_dir,
        save_summaries_secs=10)

