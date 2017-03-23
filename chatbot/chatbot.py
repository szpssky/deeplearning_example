import tensorflow as tf
import numpy as np
from chatbot import seq2seq_model
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

train_encode_vocabulary = 'train_encode_vocabulary'
train_decode_vocabulary = 'train_decode_vocabulary'


def read_vocabulary(input_file):
    tmp_vocab = []
    with open(input_file, "r") as f:
        tmp_vocab.extend(f.readlines())
    tmp_vocab = [line.strip() for line in tmp_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(tmp_vocab)])
    return vocab, tmp_vocab


vocab_en, _, = read_vocabulary(train_encode_vocabulary)
_, vocab_de, = read_vocabulary(train_decode_vocabulary)



buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
size = 512
layer_num = 4
max_gradient_norm = 5
batch_size = 1
learning_rate = 0.5
learning_rate_decay_factor = 0.98

model = seq2seq_model.Seq2SeqModel(6000, 6000, buckets, size, layer_num, max_gradient_norm, batch_size,
                                       learning_rate, learning_rate_decay_factor, use_lstm=True,
                                       forward_only=False)


model.batch_size = 1

with tf.Session() as sess:
    # 恢复前一次训练
    ckpt = tf.train.get_checkpoint_state('variables')
    if ckpt != None:
        print(ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("没找到模型")

    while True:
        input_string = input('我 : ')
        # 退出
        if input_string == 'quit':
            exit()

        input_string_vec = []
        for words in input_string.strip():
            input_string_vec.append(vocab_en.get(words, UNK_ID))
        bucket_id = min([b for b in range(len(buckets)) if buckets[b][0] > len(input_string_vec)])
        encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(input_string_vec, [])]},
                                                                         bucket_id)
        _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
        if EOS_ID in outputs:
            outputs = outputs[:outputs.index(EOS_ID)]

        response = "".join([tf.compat.as_str(vocab_de[output]) for output in outputs])
        print('AI : ' + response)