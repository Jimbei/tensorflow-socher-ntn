import params
import scipy.io as sio
import numpy as np
import csv
import random
import tensorflow as tf


# from tensorflow.python.client import device_lib


# TODO: why do we need initEmbed?
# This is a kind of pre-trained model used for initial step
# TODO: what is the content of initEmbed?
# initEmbed is a dictionary stores initial information for training


def write_words(words):
    file_path = '../data/Wordnet/additionalFiles/words.csv'

    rand_num = random.randint(0, words.shape[1])

    print('Writing file ...')
    with open(file_path, 'w', newline='') as f:
        for i in range(words.shape[1]):
            writer = csv.writer(f)

            if rand_num == i:
                print(words[0][i])

            writer.writerows([words[0][i]])

    print('Finish writing words ...')


def logging_device_placement():
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='b')
    c = tf.matmul(a, b)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    print(sess.run(c))


def get_list_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def main():
    learning_rate = 0.001
    regularization = 0.0001
    num_iter = 500

    with tf.Graph().as_default():
        training_model = training(loss(inference(), regularization), learning_rate)
        with tf.Session() as sess:
            init = tf.initialize_all_variables()


def read_file():
    print('Loading data ...')
    entity_words = []
    with open('../data/Wordnet/additionalFiles/indices.csv', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            entity_words.append(row)

    entity_words = [map(int, row) for row in entity_words]
    print('Loading is complete!')

    print('100th element is ' + str(entity_words[99]))
    print('100th element is ' + str(list(entity_words[99])))


def load_data(path_file, datatype=1):
    with open(path_file, 'r', newline='') as f:
        content = csv.reader(f)
        indices = [list(map(type(datatype), row)) for row in content]
    print('Finish loading indices.csv\n')
    return indices


def so():
    a = tf.constant([1, 2], name='a')
    b = tf.constant([3, 4], name='b')
    c = tf.constant([1, 3], name='c')

    d = tf.add_n([a, b])
    e = tf.pow(d, c)

    printout_d = tf.Print(d, [a, b, d], 'Input for d and the result: ')
    printout_e = tf.Print(e, [d, c, e], 'Input for e and the result: ')

    with tf.Session() as sess:
        sess.run([d, e])
        printout_d.eval()
        printout_e.eval()


def lab():
    num_relations = 11

    sess = tf.InteractiveSession()
    batch_placeholders = [tf.placeholder(tf.int32, shape=(None, 3), name='batch_xx' + str(i)) for i in
                          range(num_relations)]
    e1v = tf.transpose(tf.squeeze(tf.gather(ent_embed, e1, name='e1v' + str(r)), [1]))
    e2v = tf.transpose(tf.squeeze(tf.gather(ent_embed, e2, name='e2v' + str(r)), [1]))
    e3v = tf.transpose(tf.squeeze(tf.gather(ent_embed, e3, name='e3v' + str(r)), [1]))

    for r in range(num_relations):
        e1, e2, e3 = tf.split(1, 3, tf.cast(batch_placeholders[r], tf.int32))
        print(e1.eval())

    sess.close()


def inference():
    path_file_1 = '../data/Wordnet/additionalFiles/indices.csv'
    path_file_2 = '../data/Wordnet/additionalFiles/wordvecs.csv'
    indices = load_data(path_file_1)
    wordvecs = load_data(path_file_2, 1.0)

    num_relations = 11
    slice_size = 3  # slice_size = k, the depth of tensor
    embedding_size = 100
    predictions = []
    preactivation_pos = []
    preactivation_neg = []

    # defining graph
    print('Defining graph')
    batch_placeholders = \
        [tf.placeholder(tf.int32, shape=(None, 3), name='batch_xx' + str(i)) for i in range(num_relations)]

    ten_variables = tf.Variable(wordvecs, name='my_variable')
    W = [tf.Variable(tf.truncated_normal([embedding_size, embedding_size, slice_size])) for i in range(num_relations)]
    V = [tf.Variable(tf.zeros([slice_size, 2 * embedding_size])) for i in range(num_relations)]
    b = [tf.Variable(tf.zeros([slice_size, 1])) for i in range(num_relations)]
    U = [tf.Variable(tf.ones([1, slice_size])) for i in range(num_relations)]

    ten_indices = [tf.constant(e) for e in indices]
    ten_wordvecs = tf.stack([tf.reduce_mean(tf.gather(ten_variables, index)) for index in ten_indices])

    for r in range(num_relations):
        e1, e2, e3 = tf.split(1, 3, tf.cast(batch_placeholders[r], tf.int32))
        e1v = tf.transpose(tf.squeeze(tf.gather(ten_wordvecs, e1, name='e1v' + str(r)), [1]))
        e2v = tf.transpose(tf.squeeze(tf.gather(ten_wordvecs, e2, name='e2v' + str(r)), [1]))
        e3v = tf.transpose(tf.squeeze(tf.gather(ten_wordvecs, e3, name='e3v' + str(r)), [1]))

        e1v_pos = e1v
        e2v_pos = e2v
        e1v_neg = e1v
        e2v_neg = e3v
        num_rel_r = tf.expand_dims(tf.shape(e1v_pos)[1], 0)

        for slice_ in range(slice_size):
            preactivation_pos.append(tf.reduce_sum(e1v_pos * tf.matmul(W[r][:, :, slice_], e2v_pos), 0))
            preactivation_neg.append(tf.reduce_sum(e1v_neg * tf.matmul(W[r][:, :, slice_], e2v_neg), 0))

        preactivation_pos = tf.stack(preactivation_pos)
        preactivation_neg = tf.stack(preactivation_neg)

        temp2_pos = tf.matmul(V[r], tf.concat(0, [e1v_pos, e2v_pos]))
        temp2_neg = tf.matmul(V[r], tf.concat(0, [e1v_neg, e2v_neg]))

        preactivation_pos = preactivation_pos + temp2_pos + b[r]
        preactivation_neg = preactivation_neg + temp2_neg + b[r]

        activation_pos = tf.tanh(preactivation_pos)
        activation_neg = tf.tanh(preactivation_neg)

        score_pos = tf.reshape(tf.matmul(U[r], activation_pos), num_rel_r)
        score_neg = tf.reshape(tf.matmul(U[r], activation_neg), num_rel_r)
        if not False:
            predictions.append(tf.pack([score_pos, score_neg]))
        else:
            predictions.append(tf.pack([score_pos, tf.reshape(label_placeholders[r], num_rel_r)]))

    return predictions


def loss(predictions, regularization):
    print("Beginning building loss")
    temp1 = tf.maximum(tf.sub(predictions[1, :], predictions[0, :]) + 1, 0)
    temp1 = tf.reduce_sum(temp1)

    temp2 = tf.sqrt(sum([tf.reduce_sum(tf.square(var)) for var in tf.trainable_variables()]))

    temp = temp1 + (regularization * temp2)

    return temp


def training(loss, learningRate=0.001):
    print("Beginning building training")

    return tf.train.AdagradOptimizer(learningRate).minimize(loss)


if __name__ == '__main__':
    print('Hello World\n')
    lab()
    exit('exit main()')
