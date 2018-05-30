import tensorflow as tf
import numpy as np
import params


def hypothesis(data_plah,
               label_plah,
               entity_indices,
               n_relations,
               E, W, V, b, U,
               mode):
    print('Get corresponding vectors for the entities')
    tensor_entity_indices = [tf.constant(entity_i) - 1 for entity_i in entity_indices]
    word_vecs = tf.stack([tf.reduce_mean(tf.gather(E, i), 0) for i in tensor_entity_indices])

    print('Shape of entity_indices: {}'.format(np.array(entity_indices).shape))
    print('Shape of tensor_entity_indices: {}'.format(np.array(tensor_entity_indices).shape))
    print('Shape of word_vecs: ' + str(word_vecs.get_shape()))

    score_values = list()

    for r in range(n_relations):
        e1, e2, e3 = tf.split(tf.cast(data_plah[r], tf.int32), 3, axis=1)
        e1v = tf.transpose(tf.squeeze(tf.gather(word_vecs, e1, name='e1v' + str(r)), [1]))
        e2v = tf.transpose(tf.squeeze(tf.gather(word_vecs, e2, name='e2v' + str(r)), [1]))
        e3v = tf.transpose(tf.squeeze(tf.gather(word_vecs, e3, name='e3v' + str(r)), [1]))

        e1v_pos = e1v
        e2v_pos = e2v
        e1v_neg = e1v
        e2v_neg = e3v
        num_rel_r = tf.expand_dims(tf.shape(e1v_pos)[1], 0)

        preactivation_pos = list()
        preactivation_neg = list()

        # =====================================================================
        # print("Starting preactivation funcs")
        for i in range(params.slice_size):
            preactivation_pos.append(tf.reduce_sum(e1v_pos * tf.matmul(W[r][:, :, i], e2v_pos), 0))
            preactivation_neg.append(tf.reduce_sum(e1v_neg * tf.matmul(W[r][:, :, i], e2v_neg), 0))
        # =====================================================================

        # =====================================================================
        preactivation_pos = tf.stack(preactivation_pos)
        preactivation_neg = tf.stack(preactivation_neg)

        temp2_pos = tf.matmul(V[r], tf.concat([e1v_pos, e2v_pos], 0))
        temp2_neg = tf.matmul(V[r], tf.concat([e1v_neg, e2v_neg], 0))
        # =====================================================================

        # =====================================================================
        preactivation_pos = preactivation_pos + temp2_pos + b[r]
        preactivation_neg = preactivation_neg + temp2_neg + b[r]
        # =====================================================================

        activation_pos = tf.tanh(preactivation_pos)
        activation_neg = tf.tanh(preactivation_neg)

        score_pos = tf.reshape(tf.matmul(U[r], activation_pos), num_rel_r)
        score_neg = tf.reshape(tf.matmul(U[r], activation_neg), num_rel_r)
        if not mode:
            score_values.append(tf.stack([score_pos, score_neg]))
        else:
            score_values.append(tf.stack([score_pos, tf.reshape(label_plah[r], num_rel_r)]))

    score_values = tf.concat(score_values, 1)

    return score_values
