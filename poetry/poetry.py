import collections
import numpy as np
import tensorflow as tf
poetry_file = 'poetry.txt'
poetrys = []

with open(poetry_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            titile, content = line.strip().split(':')
            content = content.replace(' ', '')
            if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content:
                continue
            if len(content) < 5 or len(content) > 79:
                continue
            content = '[' + content + ']'
            poetrys.append(content)
        except Exception as e:
            pass

poetrys = sorted(poetrys, key=lambda line: len(line))
print('poetry total:' + str(len(poetrys)))

all_words = []
for poetry in poetrys:
    all_words += [word for word in poetry]
counter = collections.Counter(all_words)

count_pairs = sorted(counter.items(), key=lambda x: -x[1])
# print(count_pairs)
words, _ = zip(*count_pairs)
# print('=====')
# print(len(words))

words = words[:len(words)] + (' ',)

word_num_map = dict(zip(words, range(len(words))))
# print(word_num_map)


# 把字转换为ID
to_num = lambda word: word_num_map.get(word, len(words))
# 把诗转换为向量
poetrys_vector = [list(map(to_num, poetry)) for poetry in poetrys]

batch_size = 1
n_chunk = len(poetrys_vector) // batch_size

x_batches = []
y_batches = []

for i in range(n_chunk):
    start_index = i * batch_size
    end_index = start_index + batch_size
    batches = poetrys_vector[start_index:end_index]
    # 取64首诗中最长的诗为length
    length = max(map(len, batches))
    xdata = np.full((batch_size, length), word_num_map[' '], np.int32)
    for row in range(batch_size):
        xdata[row, :len(batches[row])] = batches[row]
    ydata = np.copy(xdata)
    ydata[:, :-1] = xdata[:, 1:]

    x_batches.append(xdata)
    y_batches.append(ydata)
    # [功成身退]/// 功成身退]////

with tf.name_scope('input_data'):
    input_data = tf.placeholder(tf.int32, [batch_size, None])
with tf.name_scope('label_data'):
    output_targets = tf.placeholder(tf.int32, [batch_size, None])


def neural_network(model='lstm', hidden_size=256, num_layers=2):
    if model == 'rnn':
        with tf.name_scope('rnn'):
            cell_fun = tf.contrib.rnn.BasicRNNCell
    elif model == 'gru':
        with tf.name_scope('gru'):
            cell_fun = tf.contrib.rnn.GRUCell
    elif model == 'lstm':
        with tf.name_scope('lstm'):
            cell_fun = tf.contrib.rnn.BasicLSTMCell

    cell = cell_fun(hidden_size, state_is_tuple=True)

    embedding = tf.get_variable("embedding", [len(words) + 1, hidden_size])
    inputs = tf.nn.embedding_lookup(embedding, input_data)  # input_data 1* 5     [212,3567,213,544,214]


    with tf.name_scope('rnn_layer'):
        cell = tf.contrib.rnn.DropoutWrapper(cell,0.5,0.5)
        cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
        initial_state = cell.zero_state(batch_size, tf.float32)
        outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnn_cell')
        # print('outputs.shape:'+str(outputs.get_shape))
        output = tf.reshape(outputs, [-1, hidden_size])  # n x 128


    with tf.variable_scope('rnn_cell'):
        # softmax_w,softmax_b 输出层权重参数
        softmax_w = tf.get_variable("softmax_w", [hidden_size, len(words) + 1])  # 128 x 6110
        softmax_b = tf.get_variable("softmax_b", [len(words) + 1])

    with tf.name_scope('softmax_layer'):
        logits = tf.matmul(output, softmax_w) + softmax_b
        probs = tf.nn.softmax(logits)

    return logits, last_state, probs, cell, initial_state


def train_neural_network():
    logits, last_state, _, _, _ = neural_network()
    targets = tf.reshape(output_targets, [-1])

    with tf.name_scope("loss"):
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [targets],
                                                                  [tf.ones_like(targets, dtype=tf.float32)],
                                                                  len(words))
        cost = tf.reduce_mean(loss)
        tf.summary.scalar('loss', cost)
    with tf.name_scope('learning_rate'):
        learning_rate = tf.Variable(0.0, trainable=False)
        tf.summary.scalar('learning_rate', learning_rate)


    with tf.name_scope('train'):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, tvars))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('log' + '/train', sess.graph)

        for epoch in range(106):
            sess.run(tf.assign(learning_rate, 0.002 * (0.97 ** epoch)))
            for batch in range(n_chunk):
                train_loss, _, _, summary = sess.run([cost, last_state, train_op, merged],
                                                     feed_dict={input_data: x_batches[batch],
                                                                output_targets: y_batches[batch]})
                train_writer.add_summary(summary, epoch * n_chunk + batch)
                print(epoch, batch, train_loss)
            if epoch % 7 == 0:
                saver.save(sess, 'variables/poetry.module', global_step=epoch)

train_neural_network()
# def gen_poetry():
#     def to_word(weights):
#         t = np.cumsum(weights)
#         s = np.sum(weights)
#         sample = int(np.searchsorted(t, np.random.rand(1) * s))
#         return words[sample]
#
#
#     _, last_state, probs, cell, initial_state = neural_network()
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         saver = tf.train.Saver(tf.global_variables())
#         module_file = tf.train.latest_checkpoint('variables')
#
#         saver.restore(sess, module_file)
#
#         state_ = sess.run(cell.zero_state(1, tf.float32))
#
#         x = np.array([list(map(word_num_map.get, '['))])
#
#         [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
#         word = to_word(probs_)
#         poem = ''
#         while word != ']':
#             poem += word
#             # if word == '。':
#             #     poem += '\n'
#             x = np.zeros((1, 1))
#             x[0, 0] = word_num_map[word]
#             [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
#             word = to_word(probs_)
#         return poem
#
# print(gen_poetry())