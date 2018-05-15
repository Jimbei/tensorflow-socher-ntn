import tensorflow as tf
import params
import ntn_input
import random
import numpy as np


# Inference
# Loss
# Training

# returns a (batch_size*corrupt_size, 2) vector corresponding to [g(T^i), g(T_c^i)] for all i
def inference(batch_placeholders,
              corrupt_placeholder,
              word_vecs,
              entity_indices,
              num_entities,
              num_relations,
              slice_size,
              batch_size,
              is_eval,
              label_placeholders):
    # TODO: We need to check the shapes and axes used here!
    print("Creating variables")
    embedding_size = 100

    E = tf.Variable(word_vecs)
    W = [tf.Variable(tf.truncated_normal([embedding_size, embedding_size, slice_size]))
         for _ in range(num_relations)]
    V = [tf.Variable(tf.zeros([slice_size, 2 * embedding_size]))
         for _ in range(num_relations)]
    b = [tf.Variable(tf.zeros([slice_size, 1]))
         for _ in range(num_relations)]
    U = [tf.Variable(tf.ones([1, slice_size]))
         for _ in range(num_relations)]

    print('Convert entity_indices to tf.constant')
    entity_indices = random.sample(entity_indices, 600)
    tensor_entity_indices = [tf.constant(entity_i) - 1
                             for entity_i in entity_indices]
    print("Calculate tensor_embedding_entity")
    tensor_embedding_entity = tf.stack([tf.reduce_mean(tf.gather(E, i), 0)
                                        for i in tensor_entity_indices])

    print('shape of tensor_embedding_entity: ' + str(tensor_embedding_entity.get_shape()))

    predictions = list()

    for r in range(num_relations):
        print('#relation: {}'.format(r))

        e1, e2, e3 = tf.split(tf.cast(batch_placeholders[r], tf.int32), 3, axis=1)
        e1v = tf.transpose(tf.squeeze(tf.gather(tensor_embedding_entity, e1, name='e1v' + str(r)), [1]))
        e2v = tf.transpose(tf.squeeze(tf.gather(tensor_embedding_entity, e2, name='e2v' + str(r)), [1]))
        e3v = tf.transpose(tf.squeeze(tf.gather(tensor_embedding_entity, e3, name='e3v' + str(r)), [1]))
        e1v_pos = e1v
        e2v_pos = e2v
        e1v_neg = e1v
        e2v_neg = e3v
        num_rel_r = tf.expand_dims(tf.shape(e1v_pos)[1], 0)
        preactivation_pos = list()
        preactivation_neg = list()

        # print("e1v_pos: "+str(e1v_pos.get_shape()))
        # print("W[r][:,:,slice]: "+str(W[r][:,:,0].get_shape()))
        # print("e2v_pos: "+str(e2v_pos.get_shape()))

        # print("Starting preactivation funcs")
        for i in range(slice_size):
            preactivation_pos.append(tf.reduce_sum(e1v_pos * tf.matmul(W[r][:, :, i], e2v_pos), 0))
            preactivation_neg.append(tf.reduce_sum(e1v_neg * tf.matmul(W[r][:, :, i], e2v_neg), 0))

        preactivation_pos = tf.stack(preactivation_pos)
        preactivation_neg = tf.stack(preactivation_neg)

        temp2_pos = tf.matmul(V[r], tf.concat([e1v_pos, e2v_pos], 0))
        temp2_neg = tf.matmul(V[r], tf.concat([e1v_neg, e2v_neg], 0))

        preactivation_pos = preactivation_pos + temp2_pos + b[r]
        preactivation_neg = preactivation_neg + temp2_neg + b[r]

        activation_pos = tf.tanh(preactivation_pos)
        activation_neg = tf.tanh(preactivation_neg)

        score_pos = tf.reshape(tf.matmul(U[r], activation_pos), num_rel_r)
        score_neg = tf.reshape(tf.matmul(U[r], activation_neg), num_rel_r)
        if not is_eval:
            predictions.append(tf.stack([score_pos, score_neg]))
        else:
            predictions.append(tf.stack([score_pos, tf.reshape(label_placeholders[r], num_rel_r)]))

    predictions = tf.concat(predictions, 1)

    return predictions


def loss(predictions, regularization):
    temp1 = tf.maximum(tf.subtract(predictions[1, :], predictions[0, :]) + 1, 0)
    temp1 = tf.reduce_sum(temp1)
    temp2 = tf.sqrt(sum([tf.reduce_sum(tf.square(var)) for var in tf.trainable_variables()]))
    temp = temp1 + (regularization * temp2)

    return temp


def training(loss, learning_rate):
    return tf.train.AdagradOptimizer(learning_rate).minimize(loss)


def eval(predictions):
    print('shape of predictions {}'.format(str(predictions.get_shape())))
    score_values, labels = tf.split(predictions, 2, axis=0)
    debug_printout = tf.Print(score_values, [score_values, labels], message='Value of inference and labels: ')

    # inference = tf.transpose(inference)
    # inference = tf.concat((1-inference), inference)
    # labels = ((tf.cast(tf.squeeze(tf.transpose(labels)), tf.int32))+1)/2
    # print("inference "+str(inference.get_shape()))
    # print("labels "+str(labels.get_shape()))
    # get number of correct labels for the logits (if prediction is top 1 closest to actual)
    # correct = tf.nn.in_top_k(inference, labels, 1)
    # cast tensor to int and return number of correct labels
    # return tf.reduce_sum(tf.cast(correct, tf.int32))
    return score_values, labels, debug_printout
