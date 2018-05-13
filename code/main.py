import csv
import tensorflow as tf
import numpy as np

import utils


# from tensorflow.python.client import device_lib


# TODO: why do we need initEmbed?
# This is a kind of pre-trained model used for initial step
# TODO: what is the content of initEmbed?
# initEmbed is a dictionary stores initial information for training
# TODO: understand tf.split()
# tensor,
# TODO: understand tf.squeeze(a_tensor, list_specified_dimension_size_1)
# Remove dimensions of size 1 from the shape of a tensor
# TODO: understand tf.expand_dims(tf.shape(e1v_pos)[1], 0)


def lab():
    sess = tf.Session()
    a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

    x, y, z = tf.split(a, 3, axis=1)

    shape_x = x.get_shape().as_list()
    shape_y = y.get_shape().as_list()
    shape_z = z.get_shape().as_list()

    print(shape_x)
    print(shape_y)
    print(shape_z)

    print_out_x = tf.Print(x, [x], summarize=x.shape[0], message='Value of x: ', name='x_value')
    print_out_y = tf.Print(y, [y], summarize=y.shape[0], message='Value of y: ', name='y_value')
    print_out_z = tf.Print(z, [z], summarize=z.shape[0], message='Value of z: ', name='z_value')

    sess.run([print_out_x, print_out_y, print_out_z])

    print(print_out_x)
    print(print_out_y)
    print(print_out_z)


def load_data(path_file, datatype=1):
    with open(path_file, 'r', newline='') as f:
        content = csv.reader(f)
        indices = [list(map(type(datatype), row)) for row in content]
    print('Finish loading indices.csv\n')
    return indices


def inference():
    num_relations = 11
    slice_size = 3  # slice_size = k, the depth of tensor
    embedding_size = 100
    predictions = []
    preactivation_pos = []
    preactivation_neg = []

    print('Loading initial parameter ...')
    wordvecs, indices = utils.load_init_embeds()
    print('Finish loading indices and wordvecs\n')

    # defining graph
    print('Defining graph')
    batch_placeholders = \
        [tf.placeholder(tf.int32, shape=(None, 3), name='batch_xx' + str(i)) for i in range(num_relations)]

    for r in range(num_relations):
        e1, e2, e3 = tf.split(1, 3, tf.cast(batch_placeholders[r], tf.int32))
        print(e1)
        print(e2)
        print(e3)

    exit()

    ten_variables = tf.Variable(wordvecs, name='my_variable', dtype=tf.float32)
    print('Converting to tensor object ...')
    ten_indices = [tf.constant(e, dtype=tf.int32) for e in indices]
    ten_wordvecs = tf.stack([tf.reduce_mean(tf.gather(ten_variables, index)) for index in ten_indices])

    W = [tf.Variable(tf.truncated_normal([embedding_size, embedding_size, slice_size])) for i in range(num_relations)]
    V = [tf.Variable(tf.zeros([slice_size, 2 * embedding_size])) for i in range(num_relations)]
    b = [tf.Variable(tf.zeros([slice_size, 1])) for i in range(num_relations)]
    U = [tf.Variable(tf.ones([1, slice_size])) for i in range(num_relations)]

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
            predictions.append(tf.stack([score_pos, score_neg]))
        else:
            predictions.append(tf.stack([score_pos, tf.reshape(label_placeholders[r], num_rel_r)]))

    exit()
    return predictions


def loss(predictions, regularization):
    print("Beginning building loss")
    temp1 = tf.maximum(tf.sub(predictions[1, :], predictions[0, :]) + 1, 0)
    temp1 = tf.reduce_sum(temp1)
    temp2 = tf.sqrt(sum([tf.reduce_sum(tf.square(var)) for var in tf.trainable_variables()]))
    temp = temp1 + (regularization * temp2)
    return temp


def training(loss, learning_rate=0.001):
    print("Beginning building training")
    return tf.train.AdagradOptimizer(learning_rate).minimize(loss)


if __name__ == '__main__':
    print('Hello World\n')
    # inference()
    lab()
    exit()
