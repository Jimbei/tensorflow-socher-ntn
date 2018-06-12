import tensorflow as tf
import params


# Inference
# Loss
# Training

# returns a (batch_size*corrupt_size, 2) vector corresponding to [g(T^i), g(T_c^i)] for all i
def hypothesis(data_plah,
               indices,
               n_relations,
               E, W, V, b, U):
    print('Convert entity_indices to tf.constant')
    indices = [tf.constant(i) - 1 for i in indices]  # one sample: indices[2] = [[[19154, 50004]]]
    entity_vecs = tf.stack([tf.reduce_mean(tf.gather(E, i), 0) for i in indices])
    print('======== DEBUG: number of entities is {}'.format(entity_vecs.get_shape()[0]))

    score_values = []

    for r in range(n_relations):
        print('#relation: {}'.format(r))
        e1, e2, e3 = tf.split(tf.cast(data_plah[r], tf.int32), 3, axis=1)
        e1v = tf.transpose(tf.squeeze(tf.gather(entity_vecs, e1, name='e1v' + str(r)), [1]))
        e2v = tf.transpose(tf.squeeze(tf.gather(entity_vecs, e2, name='e2v' + str(r)), [1]))
        e3v = tf.transpose(tf.squeeze(tf.gather(entity_vecs, e3, name='e3v' + str(r)), [1]))

        e1v_pos = e1v
        e2v_pos = e2v
        e1v_neg = e1v
        e2v_neg = e3v
        # number of triples for a single relation
        num_rel_r = tf.expand_dims(tf.shape(e1v_pos)[1], 0)

        term1_pos = []
        term1_neg = []

        # =====================================================================
        for i in range(params.slice_size):
            term1_pos.append(tf.diag_part(tf.matmul(tf.matmul(tf.transpose(e1v_pos), W[r][:, :, i]), e2v_pos)))
            term1_neg.append(tf.diag_part(tf.matmul(tf.matmul(tf.transpose(e1v_neg), W[r][:, :, i]), e1v_neg)))

        term1_pos = tf.stack(term1_pos)
        term1_neg = tf.stack(term1_neg)
        # =====================================================================

        # =====================================================================
        term2_pos = tf.matmul(V[r], tf.concat([e1v_pos, e2v_pos], 0))
        term2_neg = tf.matmul(V[r], tf.concat([e1v_neg, e2v_neg], 0))
        # =====================================================================

        # =====================================================================
        tanh_pos = tf.tanh(term1_pos + term2_pos + b[r])
        tanh_neg = tf.tanh(term1_neg + term2_neg + b[r])
        # =====================================================================

        # before:   shape(tf.matmul(U[r], tanh_pos)) == [1, 15]
        # after:    shape(tf.matmul(U[r], tanh_pos)) == [15]
        score_pos = tf.reshape(tf.matmul(U[r], tanh_pos), num_rel_r)
        score_neg = tf.reshape(tf.matmul(U[r], tanh_neg), num_rel_r)

        # shape(score_pos) == [15], shape(score_neg) == [15]
        # shape(tf.stack([score_pos, score_cor])) == [2, 15]
        score_values.append(tf.stack([score_pos, score_neg]))

    score_values = tf.concat(score_values, 1)
    foo = tf.constant(1)

    return score_values, foo


def loss(score_values, regularization):
    term1 = tf.reduce_sum(tf.maximum(tf.subtract(score_values[1, :], score_values[0, :]) + 1, 0))
    term2 = tf.sqrt(sum([tf.reduce_sum(tf.square(var)) for var in tf.trainable_variables()]))
    loss_values = term1 + (regularization * term2)

    return loss_values


def training(loss, learning_rate):
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    # return tf.train.AdagradOptimizer(learning_rate).minimize(loss)


def eval(score_values):
    score_values, labels = tf.split(score_values, 2, axis=0)
    return score_values, labels
