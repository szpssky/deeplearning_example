import tensorflow as tf
import numpy as np
import sys
from chatbot import data_utils
from chatbot import seq2seq_model
import time
import math

buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
size = 512
layer_num = 4
max_gradient_norm = 5
batch_size = 64
learning_rate = 0.5
learning_rate_decay_factor = 0.98
train_encode_vec = 'train_encode_vec'
train_decode_vec = 'train_decode_vec'
test_encode_vec = 'test_encode_vec'
test_decode_vec = 'test_decode_vec'


def readData(sourcePath, targetPath, max_size=None):
    dataset = [[] for _ in buckets]
    with tf.gfile.GFile(sourcePath, mode='r') as sourcefile:
        with tf.gfile.GFile(targetPath, mode='r') as targetfile:
            source, target = sourcefile.readline(), targetfile.readline()
            count = 0
            while source and target and (not max_size or count < max_size):
                count += 1
                if count % 100000 == 0:
                    print("  reading data line %d" % count)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(data_utils.EOS_ID)
                for buckets_id, (source_size, target_size) in enumerate(buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        dataset[buckets_id].append([source_ids, target_ids])
                        break

                source, target = sourcefile.readline(), targetfile.readline()
    return dataset


def create_model(session, forword_only):
    model = seq2seq_model.Seq2SeqModel(6000, 6000, buckets, size, layer_num, max_gradient_norm, batch_size,
                                       learning_rate, learning_rate_decay_factor, use_lstm=True,
                                       forward_only=forword_only)

    ckpt = tf.train.get_checkpoint_state("variables")
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


def train():
    with tf.Session() as sess:

        model = create_model(sess, False)
        train_set = readData(train_encode_vec, train_decode_vec)
        test_set = readData(test_encode_vec, test_decode_vec)

        train_bucket_sizes = [len(train_set[b]) for b in range(len(buckets))]

        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in range(len(train_bucket_sizes))]

        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []

        while True:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                train_set, bucket_id)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, False)
            step_time += (time.time() - start_time) / 100
            loss += step_loss / 100
            current_step += 1

            if current_step % 100 == 0:

                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                print("global step %d learning rate %.4f step-time %.2f perplexity %.2f loss %.2f"
                      % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity, loss))

                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                checkpoint_path = 'variables/catbot-model.ckpt'
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                for bucket_id in range(len(buckets)):
                    if len(test_set[bucket_id]) == 0:
                        print("  eval: empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(test_set, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                 target_weights, bucket_id, True)
                    eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
                        "inf")
                    print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                sys.stdout.flush()


train()
