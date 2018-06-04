import tensorflow as tf
import numpy as  np
import random

import ntn_input
import params

saved_model = 'Wordnet490'

relation_list = ['_has_instance',
                 '_type_of',
                 '_member_meronym',
                 '_member_holonym',
                 '_part_of',
                 '_has_part',
                 '_subordinate_instance_of',
                 '_domain_region',
                 '_synset_domain_topic',
                 '_similar_to',
                 '_domain_topic']


def my_function(data_plah,
                indices,
                r,
                E, W, V, b, U):
    indices = [tf.constant(i) - 1 for i in indices]
    entity_vecs = tf.stack([tf.reduce_mean(tf.gather(E, i), 0) for i in indices])
    
    e1, e2 = tf.split(data_plah[0], 2, axis=1)
    e1v = tf.transpose(tf.squeeze(tf.gather(entity_vecs, e1, name='e1T'), [1]))
    e2v = tf.transpose(tf.squeeze(tf.gather(entity_vecs, e2, name='e2T'), [1]))
    
    num_rel_r = tf.expand_dims(tf.shape(e1v)[1], 0)
    
    term1 = []
    term1.append(tf.diag_part(tf.matmul(tf.matmul(tf.transpose(e1v), W[r][:, :, 0]), e2v)))
    term1.append(tf.diag_part(tf.matmul(tf.matmul(tf.transpose(e1v), W[r][:, :, 1]), e2v)))
    term1.append(tf.diag_part(tf.matmul(tf.matmul(tf.transpose(e1v), W[r][:, :, 2]), e2v)))
    term1 = tf.stack(term1)
    
    tmp1 = tf.concat([e1v, e2v], 0)
    term2 = tf.matmul(V[r], tmp1)
    
    tanh_function = tf.tanh(term1 + term2 + b[r])
    
    score_values = tf.reshape(tf.matmul(U[r], tanh_function), num_rel_r)
    # debug
    tmp = score_values
    foo = tf.Print(tmp, [tmp, tf.shape(tmp)], message='======== DEBUG: ')
    # =========================================================================
    
    return score_values, foo


def load_input():
    entity_list = ntn_input.load_entities(params.data_path)
    indexing_entities = {entity_list[i]: i for i in range(len(entity_list))}
    
    triples = [['__pheasant_1', '__phasianus_colchicus_1'],
               ['__case_9', '__frame_7']]
    # pair_e = []
    # for i in range(2):
    #     pair_e = random.sample(entity_list, 2)
    
    triples = [[indexing_entities[triples[0][0]], indexing_entities[triples[0][1]]],
               [indexing_entities[triples[1][0]], indexing_entities[triples[1][1]]]]
    
    return triples


def run_production():
    print('Load initial embedding vectors')
    init_word_embeds, indices = ntn_input.load_init_embeds(params.data_path)
    print('Load input')
    triples = load_input()
    indices = [indices[triples[0][0]], indices[triples[0][1]]]
    # relation_list = ntn_input.load_relations(params.data_path)
    n_relations = len(relation_list)
    print('List of relations: {}'.format(relation_list))
    
    # define placeholder
    print('Define placeholder')
    data_plah = [tf.placeholder(tf.int32, shape=(None, 2), name='pair_entities')]
    
    # define variables
    print('Define variables')
    E = tf.Variable(init_word_embeds)
    W = [tf.Variable(tf.truncated_normal([params.embedding_size, params.embedding_size, params.slice_size]))
         for _ in range(n_relations)]
    V = [tf.Variable(tf.zeros([params.slice_size, 2 * params.embedding_size]))
         for _ in range(n_relations)]
    b = [tf.Variable(tf.zeros([params.slice_size, 1]))
         for _ in range(n_relations)]
    U = [tf.Variable(tf.ones([1, params.slice_size]))
         for _ in range(n_relations)]
    
    score_r0, foo0 = my_function(data_plah, indices, 0, E, W, V, b, U)
    score_r1, foo1 = my_function(data_plah, indices, 1, E, W, V, b, U)
    score_r2, foo2 = my_function(data_plah, indices, 2, E, W, V, b, U)
    score_r3, foo3 = my_function(data_plah, indices, 3, E, W, V, b, U)
    score_r4, foo4 = my_function(data_plah, indices, 4, E, W, V, b, U)
    score_r5, foo5 = my_function(data_plah, indices, 5, E, W, V, b, U)
    score_r6, foo6 = my_function(data_plah, indices, 6, E, W, V, b, U)
    score_r7, foo7 = my_function(data_plah, indices, 7, E, W, V, b, U)
    score_r8, foo8 = my_function(data_plah, indices, 8, E, W, V, b, U)
    score_r9, foo9 = my_function(data_plah, indices, 9, E, W, V, b, U)
    score_r10, foo10 = my_function(data_plah, indices, 10, E, W, V, b, U)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('Load checkpoint ' + saved_model)
        saver = tf.train.Saver()
        saver.restore(sess, params.output_path + saved_model + '.sess')
        print('Feed {} and {}'.format(triples[0], triples[1]))
        feed_dict = {data_plah[0]: triples}
        
        score_r0, foo0 = sess.run([score_r0, foo0], feed_dict)
        score_r1, foo1 = sess.run([score_r1, foo1], feed_dict)
        score_r2, foo2 = sess.run([score_r2, foo2], feed_dict)
        score_r3, foo3 = sess.run([score_r3, foo3], feed_dict)
        score_r4, foo4 = sess.run([score_r4, foo4], feed_dict)
        score_r5, foo5 = sess.run([score_r5, foo5], feed_dict)
        score_r6, foo6 = sess.run([score_r6, foo6], feed_dict)
        score_r7, foo7 = sess.run([score_r7, foo7], feed_dict)
        score_r8, foo8 = sess.run([score_r8, foo8], feed_dict)
        score_r9, foo9 = sess.run([score_r9, foo9], feed_dict)
        score_r10, foo10 = sess.run([score_r10, foo10], feed_dict)
        
        score_values, relations = [], []
        score_values.append(score_r0)
        score_values.append(score_r1)
        score_values.append(score_r2)
        score_values.append(score_r3)
        score_values.append(score_r4)
        score_values.append(score_r5)
        score_values.append(score_r6)
        score_values.append(score_r7)
        score_values.append(score_r8)
        score_values.append(score_r9)
        score_values.append(score_r10)
        
        score_values = np.array(score_values)
        score_values = np.transpose(score_values)
        print('\nScore values: {} has shape of {}'.format(score_values, score_values.shape))
        for values in score_values:
            values = values.tolist()
            print('comparing values {}'.format(values))
            print('max value: {} at the index: {}'.format(max(values), values.index(max(values))))
            relations.append(values.index(max(values)))
        
        print('score_r0: {}'
              '\nscore_r1: {}'
              '\nscore_r2: {}'
              '\nscore_r3: {}'
              '\nscore_r4: {}'
              '\nscore_r5: {}'
              '\nscore_r6: {}'
              '\nscore_r7: {}'
              '\nscore_r8: {}'
              '\nscore_r9: {}'
              '\nscore_r10: {}'.format(score_r0,
                                       score_r1,
                                       score_r2,
                                       score_r3,
                                       score_r4,
                                       score_r5,
                                       score_r6,
                                       score_r7,
                                       score_r8,
                                       score_r9,
                                       score_r10))
        print('relations: {} and {}'.format(relation_list[relations[0]], relation_list[relations[1]]))


if __name__ == '__main__':
    run_production()
