import tensorflow as tf
import params
import ntn_input
import random
import math


def g_function(batch_placeholders,
               corrupt_placeholder,
               word_vecs,
               entity_indices,
               num_entities,
               num_relations,
               slice_size,
               batch_size,
               is_eval,
               label_placeholders):
    print("Begin building g function:")
    # TODO: Check the shapes and axes used here!
    print("Creating variables")
    d = 100  # embedding_size
    k = slice_size
    ten_k = tf.constant([k])
    num_words = len(word_vecs)
    # TODO: does wordvecs need to change to list?
    var_E = tf.Variable(word_vecs)  # create a variable with initial values from wordvecs
    var_W = [tf.Variable(tf.truncated_normal([d, d, k])) for r in range(num_relations)]
    var_V = [tf.Variable(tf.zeros([k, 2 * d])) for r in range(num_relations)]
    var_b = [tf.Variable(tf.zeros([k, 1])) for r in range(num_relations)]
    var_U = [tf.Variable(tf.ones([1, k])) for r in range(num_relations)]

    print("Convert entity_indices to tensor_entity_indices ...")
    entity_indices = random.sample(entity_indices, 600)
    tensor_entity_indices = [tf.constant(entity_i, tf.int32) - 1 for entity_i in entity_indices]

    print("Calculate tensor_embedding_entity ...")
    tensor_embedding_entity = tf.stack([tf.reduce_mean(tf.gather(var_E, ei), 0)
                                        for ei in tensor_entity_indices])

    predictions = list()
    print("Beginning relations loop")
    for r in range(num_relations):
        print("#relations: " + str(r))
        # TODO: should the split dimension be 0 or 1?
        e1, e2, e3 = tf.split(1, 3, tf.cast(batch_placeholders[r], tf.int32))
        e1v = tf.transpose(tf.squeeze(tf.gather(tensor_embedding_entity, e1, name='e1v' + str(r)), [1]))
        e2v = tf.transpose(tf.squeeze(tf.gather(tensor_embedding_entity, e2, name='e2v' + str(r)), [1]))
        e3v = tf.transpose(tf.squeeze(tf.gather(tensor_embedding_entity, e3, name='e3v' + str(r)), [1]))
        exit()
        
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
        for slice_ in range(k):
            preactivation_pos.append(tf.reduce_sum(e1v_pos * tf.matmul(var_W[r][:, :, slice_], e2v_pos), 0))
            preactivation_neg.append(tf.reduce_sum(e1v_neg * tf.matmul(var_W[r][:, :, slice_], e2v_neg), 0))

        preactivation_pos = tf.stack(preactivation_pos)
        preactivation_neg = tf.stack(preactivation_neg)

        temp2_pos = tf.matmul(var_V[r], tf.concat(0, [e1v_pos, e2v_pos]))
        temp2_neg = tf.matmul(var_V[r], tf.concat(0, [e1v_neg, e2v_neg]))

        # print("   temp2_pos: "+str(temp2_pos.get_shape()))
        preactivation_pos = preactivation_pos + temp2_pos + var_b[r]
        preactivation_neg = preactivation_neg + temp2_neg + var_b[r]

        # print("Starting activation funcs")
        activation_pos = tf.tanh(preactivation_pos)
        activation_neg = tf.tanh(preactivation_neg)

        score_pos = tf.reshape(tf.matmul(var_U[r], activation_pos), num_rel_r)
        score_neg = tf.reshape(tf.matmul(var_U[r], activation_neg), num_rel_r)
        # print("score_pos: "+str(score_pos.get_shape()))
        if not is_eval:
            predictions.append(tf.stack([score_pos, score_neg]))
        else:
            predictions.append(tf.stack([score_pos, tf.reshape(label_placeholders[r], num_rel_r)]))

    predictions = tf.concat(1, predictions)

    return predictions


def loss(predictions, regularization):
    print("Beginning building loss")
    temp1 = tf.maximum(tf.sub(predictions[1, :], predictions[0, :]) + 1, 0)
    temp1 = tf.reduce_sum(temp1)

    temp2 = tf.sqrt(sum([tf.reduce_sum(tf.square(var)) for var in tf.trainable_variables()]))

    temp = temp1 + (regularization * temp2)

    return temp


def training(loss, learningRate):
    print("Beginning building training")

    return tf.train.AdagradOptimizer(learningRate).minimize(loss)


def eval(predictions):
    print("predictions " + str(predictions.get_shape()))
    inference, labels = tf.split(0, 2, predictions)
    # inference = tf.transpose(inference)
    # inference = tf.concat((1-inference), inference)
    # labels = ((tf.cast(tf.squeeze(tf.transpose(labels)), tf.int32))+1)/2
    # print("inference "+str(inference.get_shape()))
    # print("labels "+str(labels.get_shape()))
    # get number of correct labels for the logits (if prediction is top 1 closest to actual)
    # correct = tf.nn.in_top_k(inference, labels, 1)
    # cast tensor to int and return number of correct labels
    # return tf.reduce_sum(tf.cast(correct, tf.int32))
    return inference, labels
