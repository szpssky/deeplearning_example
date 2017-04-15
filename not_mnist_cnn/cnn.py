import pickle
import tensorflow as tf
import numpy as np


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1))
            / predictions.shape[0])


with open('notMNIST.pickle', 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']

batch_size = 1
num_labels = 10
num_channnels = 1
depth = 32
image_size = 28
path_size = 5

train_dataset = np.reshape(train_dataset, (-1, image_size, image_size, num_channnels))
train_labels = (np.arange(num_labels) == train_labels[:, None]).astype(np.float32)

test_dataset = np.reshape(test_dataset, (-1, image_size, image_size, num_channnels))
test_labels = (np.arange(num_labels) == test_labels[:, None]).astype(np.float32)

print(train_dataset.shape)
print(train_labels.shape)

with tf.name_scope('input_data'):
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channnels))
with tf.name_scope('labels'):
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

tf_test_dataset = tf.constant(test_dataset)


def cnn(data):
    with tf.name_scope('weights'):
        weights = {'w_conv1': tf.Variable(tf.random_normal([path_size, path_size, num_channnels, depth])),
                   'w_conv2': tf.Variable(tf.random_normal([path_size, path_size, depth, 64])),
                   'w_fc1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
                   'w_fc2': tf.Variable(tf.random_normal([1024, 1024])),
                   'out': tf.Variable(tf.random_normal([1024, num_labels]))}
    with tf.name_scope('biases'):
        biases = {
            'b_conv1': tf.Variable(tf.random_normal([32])),
            'b_conv2': tf.Variable(tf.random_normal([64])),
            'b_fc1': tf.Variable(tf.random_normal([1024])),
            'b_fc2': tf.Variable(tf.random_normal([1024])),
            'b_out': tf.Variable(tf.random_normal([num_labels]))
        }

    with tf.name_scope('conv1'):
        conv1 = tf.nn.relu(
            tf.nn.conv2d(data, weights['w_conv1'], [1, 1, 1, 1], padding="SAME", use_cudnn_on_gpu=False, name="conv1") + \
            biases['b_conv1'])

    # print("conv1", conv1.shape)
    with tf.name_scope('pool1'):
        pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME", name='pool1')
    # print("pool1", pool1.shape)

    with tf.name_scope('conv2'):
        conv2 = tf.nn.relu(
            tf.nn.conv2d(pool1, weights['w_conv2'], [1, 1, 1, 1], padding="SAME", use_cudnn_on_gpu=False,
                         name="conv2") + biases['b_conv2'])

    # print("conv2", conv2.shape)
    with tf.name_scope('pool2'):
        pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME", name="pool2")

    # print("pool2", pool2.shape)

    # print("reshape", tf.reshape(pool2, [-1, 7 * 7 * 64]).shape)
    with tf.name_scope('fc_1'):
        fc_1 = tf.nn.relu(tf.matmul(tf.reshape(pool2, [-1, 7 * 7 * 64]), weights['w_fc1']) + biases['b_fc1'])

    # print("fc1", fc_1.shape)

    with tf.name_scope('fc_2'):
        fc_2 = tf.nn.relu(tf.matmul(fc_1, weights['w_fc2']) + biases['b_fc2'])

    # print("fc2", fc_2.shape)

    with tf.name_scope('output'):
        output = tf.matmul(fc_2, weights['out']) + biases['b_out']

    return output


logits = cnn(tf_train_dataset)
predict = tf.nn.softmax(logits)

test_prediction = tf.nn.softmax(cnn(tf_test_dataset))

# print(tf_train_dataset.shape)
# print(tf_train_labels.shape)
# print(logits.shape)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    tf.summary.scalar('loss', loss)

with tf.name_scope('training'):
    train_step = tf.train.AdamOptimizer(0.05).minimize(loss)


# tvars = tf.trainable_variables()
# grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
# optimizer = tf.train.AdamOptimizer()
# train_op = optimizer.apply_gradients(zip(grads, tvars))


def train():

    with tf.Session() as sess:

        saver = tf.train.Saver(tf.global_variables())

        ckpt = tf.train.get_checkpoint_state("variables")
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            sess.run(tf.global_variables_initializer())


        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('log' + '/train', sess.graph)

        epoch = 0
        while True:
            offset = (epoch * batch_size) % (train_labels.shape[0] - batch_size)

            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]

            _, l, prediction, summary = sess.run([train_step, loss, predict, merged],
                                                 feed_dict={tf_train_dataset: batch_data,
                                                            tf_train_labels: batch_labels})
            train_writer.add_summary(summary, epoch)

            epoch += 1
            if epoch % 1000 == 0:
                print("save variables")
                saver.save(sess, 'variables/poetry.module', global_step=epoch)

            if epoch % 5000==0:
                print("Epoch:",epoch)
                print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
                print("===================================================")



train()
