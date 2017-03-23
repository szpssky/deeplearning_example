import numpy as np
import tensorflow as tf
import random
import pickle

# with open('corpus_ask.pick', 'rb') as f:
#     ask = pickle.load(f)
# with open('corpus_ans.pick','rb') as f:
#     ans = pickle(f)
with open('conv.pick','rb') as f:
    convs = pickle.load(f)

ask=[]
ans=[]
for conv in convs:
    if len(conv) % 2 != 0:
        conv = conv[:-1]
    for i in range(len(conv)):
        if i % 2 == 0:
            ask.append(conv[i])
        else:
            ans.append(conv[i])


# print(ask[:10])
# print(ans[:10])
def convert_seq2seq_files(questions, answers, TESTSET_SIZE=8000):
    # 创建文件
    train_enc = open('train.enc', 'w')  # 问
    train_dec = open('train.dec', 'w')  # 答
    test_enc = open('test.enc', 'w')  # 问
    test_dec = open('test.dec', 'w')  # 答

    # 选择20000数据作为测试数据
    test_index = random.sample([i for i in range(len(questions))], TESTSET_SIZE)

    for i in range(len(questions)):
        if i in test_index:
            test_enc.write(questions[i] + '\n')
            test_dec.write(answers[i] + '\n')
        else:
            train_enc.write(questions[i] + '\n')
            train_dec.write(answers[i] + '\n')
        if i % 1000 == 0:
            print(len(range(len(questions))), '处理进度:', i)

    train_enc.close()
    train_dec.close()
    test_enc.close()
    test_dec.close()



# -----------------------------------------

PAD = "__PAD__"
GO = "__GO__"
EOS = "__EOS__"  # 对话结束
UNK = "__UNK__"  # 标记未出现在词汇表中的字符
START_VOCABULART = [PAD, GO, EOS, UNK]
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size):
    if not tf.gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
        vocab = {}
        with tf.gfile.GFile(data_path, mode="rb") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print("  processing line %d" % counter)
                tokens = [word for word in line.strip()]
                for word in tokens:
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1

            vocab_list = START_VOCABULART + sorted(vocab, key=vocab.get, reverse=True)
            print(len(vocab_list))
            if len(vocab_list) > max_vocabulary_size:
                vocab_list = vocab_list[:max_vocabulary_size]
            with tf.gfile.GFile(vocabulary_path, mode='wb') as f:
                for w in vocab_list:
                    f.write(w + '\n')


def initialize_vocabulary(vocabulary_path):
    if tf.gfile.Exists(vocabulary_path):
        rev_vocab = []
        with tf.gfile.GFile(vocabulary_path, mode='rb') as f:
            rev_vocab.extend(f.readlines())

        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab


def convertToVec(data_path, target_path, vocabulary_path):
    if not tf.gfile.Exists(target_path):
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with tf.gfile.GFile(data_path, mode='rb') as data_file:
            with tf.gfile.GFile(target_path, mode='w') as vec_file:
                count = 0
                for line in data_file:
                    line_vec = []
                    count += 1
                    if count % 100000 == 0:
                        print("  tokenizing line %d" % count)
                    for word in line.strip():
                        line_vec.append(vocab.get(word, UNK_ID))

                    vec_file.write(" ".join([str(num) for num in line_vec]) + '\n')


convert_seq2seq_files(ask, ans,8000)
create_vocabulary("train_encode_vocabulary","train.enc",6000)
create_vocabulary("train_decode_vocabulary","train.dec",6000)

convertToVec("train.enc","train_encode_vec","train_encode_vocabulary")
convertToVec("train.dec","train_decode_vec","train_decode_vocabulary")
convertToVec("test.enc","test_encode_vec","train_encode_vocabulary")
convertToVec("test.dec","test_decode_vec","train_decode_vocabulary")