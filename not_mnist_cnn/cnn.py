import pickle
import tensorflow as tf
import numpy as np

with open('notMNIST.pickle', 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']

batch_size = 16
num_labels = 10
num_channnels = 1
depth = 32
image_size = 28
path_size = 5

train_dataset = np.reshape(train_dataset, (-1, image_size, image_size, num_channnels))
train_labels = (np.arange(num_labels) == train_labels[:, None]).astype(np.float32)
print(test_dataset.shape)
test_dataset = test_dataset[0:1000]
test_labels = test_labels[0:1000]

test_dataset = np.reshape(test_dataset, (-1, image_size, image_size, num_channnels))
test_labels = (np.arange(num_labels) == test_labels[:, None]).astype(np.float32)

print(train_dataset.shape)
print(train_labels.shape)

with tf.name_scope('input_data'):
    tf_train_dataset = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channnels))
with tf.name_scope('labels'):
    tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_labels))

global_step = tf.Variable(0, trainable=False)


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1))
            / predictions.shape[0])


with tf.name_scope('weights'):
    weights = {'w_conv1': tf.Variable(tf.random_normal([path_size, path_size, num_channnels, depth], stddev=0.1)),
               'w_conv2': tf.Variable(tf.random_normal([path_size, path_size, depth, 64], stddev=0.1)),
               'w_fc1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024], stddev=0.1)),
               'w_fc2': tf.Variable(tf.random_normal([1024, 1024], stddev=0.1)),
               'out': tf.Variable(tf.random_normal([1024, num_labels], stddev=0.1))}
with tf.name_scope('biases'):
    biases = {
        'b_conv1': tf.Variable(tf.constant(1.0, shape=[32])),
        'b_conv2': tf.Variable(tf.constant(1.0, shape=[64])),
        'b_fc1': tf.Variable(tf.constant(1.0, shape=[1024])),
        'b_fc2': tf.Variable(tf.constant(1.0, shape=[1024])),
        'b_out': tf.Variable(tf.constant(1.0, shape=[num_labels]))
    }


def cnn(data):
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

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    loss = loss + 0.000001 * (
        tf.nn.l2_loss(weights['w_fc1']) + tf.nn.l2_loss(weights['w_fc2']) + tf.nn.l2_loss(
            biases['b_fc1'] + tf.nn.l2_loss(biases['b_fc2'])) + tf.nn.l2_loss(weights['out']) + tf.nn.l2_loss(
            biases['b_out']))

    tf.summary.scalar('loss', loss)

with tf.name_scope('training'):
    starter_learning_rate = 0.05
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 10000, 0.96, staircase=True)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predict, 1), tf.argmax(tf_train_labels, 1)), tf.float32))
    tf.summary.scalar('Train Accuary', acc)


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
            print("parameters initializer .")
            sess.run(tf.global_variables_initializer())

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('log' + '/train', sess.graph)

        Iterator = 0
        while True:
            offset = (Iterator * batch_size) % (train_labels.shape[0] - batch_size)

            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]

            _, l, prediction, summary = sess.run([train_step, loss, predict, merged],
                                                 feed_dict={tf_train_dataset: batch_data,
                                                            tf_train_labels: batch_labels})
            train_writer.add_summary(summary, Iterator)

            Iterator += 1
            if Iterator % 1000 == 0:
                print("save variables")
                saver.save(sess, 'variables/poetry.module', global_step=Iterator)

            if Iterator % 1000 == 0:
                print("Iterator:", Iterator)
                print("loss:", l)
                print('Test accuracy: %.1f%%' % accuracy(predict.eval({tf_train_dataset: test_dataset}), test_labels))
                print("===================================================")


train()
