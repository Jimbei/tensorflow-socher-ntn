import tensorflow as tf
import ntn_input
import params
import datamanipulation as dp
import random
import numpy as np


def hypothesis(data_placho,
               label_placho,
               entity_indices,
               n_relations,
               E, W, V, b, U,
               is_eval=False):
    print('Get corresponding vectors for the entities')
    tensor_entity_indices = [tf.constant(entity_i) - 1 for entity_i in entity_indices]
    word_vecs = tf.stack([tf.reduce_mean(tf.gather(E, i), 0) for i in tensor_entity_indices])
    
    print('Shape of word_vecs: ' + str(word_vecs.get_shape()))
    
    score_values = list()
    
    for r in range(n_relations):
        e1, e2, e3 = tf.split(tf.cast(data_placho[r], tf.int32), 3, axis=1)
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
        if not is_eval:
            score_values.append(tf.stack([score_pos, score_neg]))
        else:
            score_values.append(tf.stack([score_pos, tf.reshape(label_placho[r], num_rel_r)]))
    
    score_values = tf.concat(score_values, 1)
    
    return score_values


def loss_function(score_values, regularization):
    term1 = tf.reduce_sum(tf.maximum(tf.subtract(score_values[1, :], score_values[0, :]) + 1, 0))
    term2 = tf.sqrt(sum([tf.reduce_sum(tf.square(var)) for var in tf.trainable_variables()]))
    return term1 + regularization * term2


def fill_feed_dict(data_plah, label_plah, corrupt_plah,
                   triples, batch_size, corrupt_size, n_relations, n_entities):
    feed_dict = {corrupt_plah: [False and np.random.random() > 0.5]}
    
    sorted_triples = [[t for t in triples if r == t[1]] for r in range(n_relations)]
    
    batch_size = int(batch_size / n_relations)
    random_data = []
    for t in sorted_triples:
        if len(t) > batch_size:
            t = random.sample(t, batch_size)
        random_data.append(t)
    
    corrupting_data = []
    for r in random_data:
        for T in r:
            for i in range(corrupt_size):
                corrupting_data.append([T[0], T[1], T[2], random.randint(0, n_entities - 1)])
    
    relation_batches = [[] for _ in range(n_relations)]
    for e1, r, e2, e3 in corrupting_data:
        relation_batches[r].append([e1, e2, e3])

    for i in range(len(data_plah)):
        feed_dict[data_plah[i]] = relation_batches[i]
        feed_dict[label_plah[i]] = [[0.0] for _ in range(len(relation_batches[i]))]

    return feed_dict


# TODO how do they feed data
def run_training():
    print('Load initial embedding word vectors')
    init_word_vecs, entity_indices = ntn_input.load_init_embeds(params.DATA_DIR)
    print('Load training data')
    triples = ntn_input.load_training_data(params.DATA_DIR)
    entity_list = ntn_input.load_entities(params.DATA_DIR)
    print('Index data')
    triples, fil_entity_indices, n_entities, n_relations = dp.index_data(triples, entity_list, entity_indices)
    
    batch_size = int(len(triples) / 3)
    
    # define placeholder
    data_plah = [tf.placeholder(tf.int32, shape=(None, 3), name='batch_' + str(i))
                 for i in range(n_relations)]
    label_plah = [tf.placeholder(tf.float32, shape=(None, 1), name='label_' + str(i))
                  for i in range(n_relations)]
    corrupt_plah = tf.placeholder(tf.bool, shape=(1))
    
    # define variables
    E = tf.Variable(init_word_vecs)
    W = [tf.Variable(tf.truncated_normal([params.embedding_size, params.embedding_size, params.slice_size]))
         for _ in range(n_relations)]
    V = [tf.Variable(tf.zeros([params.slice_size, 2 * params.embedding_size]))
         for _ in range(n_relations)]
    b = [tf.Variable(tf.zeros([params.slice_size, 1]))
         for _ in range(n_relations)]
    U = [tf.Variable(tf.ones([1, params.slice_size]))
         for _ in range(n_relations)]
    
    # define hypothesis
    score_values = hypothesis(data_plah,
                              label_plah,
                              fil_entity_indices,
                              n_relations,
                              E, W, V, b, U,
                              False)
    # define loss function
    loss = loss_function(score_values, params.regularization)
    
    # initialize variables
    optimizer = tf.train.AdagradOptimizer(params.learning_rate).minimize(loss)
    
    # execute graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for i in range(1, params.n_iters):
            feed_dict = fill_feed_dict(data_plah, label_plah, corrupt_plah,
                                       triples, batch_size, params.corrupt_size, n_relations, n_entities)
            
            _, loss_value = sess.run([optimizer, loss], feed_dict=feed_dict)

        E, W, V, b, U = sess.run([E, W, V, b, U])
