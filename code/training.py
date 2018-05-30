import tensorflow as tf
import params
import datamanipulation as dp
import random
import numpy as np
import hypothesis


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
    init_word_vecs, entity_indices = dp.load_init_embeds(params.DATA_DIR)
    print('Load training data')
    triples = dp.load_triples(params.DATA_DIR, 0)
    entity_list = dp.load_entities(params.DATA_DIR)
    print('Index data')
    triples, fil_entity_indices, n_entities, n_relations = dp.index_data(triples, entity_list, entity_indices)

    batch_size = int(len(triples) / 3)

    # define placeholder
    print('Define placeholders')
    data_plah = [tf.placeholder(tf.int32, shape=(None, 3), name='batch_' + str(i))
                 for i in range(n_relations)]
    label_plah = [tf.placeholder(tf.float32, shape=(None, 1), name='label_' + str(i))
                  for i in range(n_relations)]
    corrupt_plah = tf.placeholder(tf.bool, shape=(1))

    # define variables
    print('Define variables')
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
    print('Define hypothesis')
    score_values = hypothesis.hypothesis(data_plah,
                                         label_plah,
                                         fil_entity_indices,
                                         n_relations,
                                         E, W, V, b, U,
                                         False)
    # define loss function
    print('Define loss function')
    loss = loss_function(score_values, params.regularization)

    # initialize variables
    print('Define optimizer')
    optimizer = tf.train.AdagradOptimizer(params.learning_rate).minimize(loss)

    # execute graph
    with tf.Session() as sess:
        print('Initialize variables')
        sess.run(tf.global_variables_initializer())

        for i in range(1, params.n_iters):
            print('Feed data')
            feed_dict = fill_feed_dict(data_plah, label_plah, corrupt_plah,
                                       triples, batch_size, params.corrupt_size, n_relations, n_entities)

            _, loss_value = sess.run([optimizer, loss], feed_dict=feed_dict)
            print('Epoch {}: {}'.format(i, loss_value))

        E, W, V, b, U = sess.run([E, W, V, b, U])
