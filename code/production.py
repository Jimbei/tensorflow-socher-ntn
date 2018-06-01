import tensorflow as tf
import ntn_input
import params
import random

saved_model = 'Wordnet490'


def load_input():
    entity_list = ntn_input.load_entities(params.data_path)
    indexing_entities = {entity_list[i]: i for i in range(len(entity_list))}
    
    pair_e = ['__sneezeweed_1', '__choeronycteris_mexicana_1']
    # pair_e = []
    # for i in range(2):
    #     pair_e = random.sample(entity_list, 2)
    
    indexing_e1 = indexing_entities[pair_e[0]]
    indexing_e2 = indexing_entities[pair_e[1]]
    return [indexing_e1, indexing_e2]


def predict_relation(e1, e2, entity_indices, n_relations, E, W, V, b, U):
    entity_indices = [tf.constant(i) - 1 for i in entity_indices]
    init_word_embeds = tf.stack([tf.reduce_mean(tf.gather(E, i), 0) for i in entity_indices])
    
    predictions = []
    for r in range(n_relations):
        
        e1 = tf.transpose(tf.squeeze(tf.gather(init_word_embeds, e1, name='e1_' + str(r)), [0]))
        e2 = tf.transpose(tf.squeeze(tf.gather(init_word_embeds, e2, name='e2_' + str(r)), [0]))
        print('Shape of e1 {}'.format(e1.get_shape()))
        print('Shape of e2 {}'.format(e2.get_shape()))
        
        term1 = []
        for i in range(params.slice_size):
            term1.append(tf.reduce_sum(e1 * tf.matmul(W[r][:, :, i], e2), 0))
        term1 = tf.stack(term1)
        
        term2 = tf.matmul(V[r], tf.concat([e1, e2], 0))
        
        tanh_function = tf.tanh(term1 + term2 + b[r])
        
        score_value = tf.matmul(U[r], tanh_function)
        
        predictions.append(score_value)
    
    return predictions.index(max(predictions))


def run_production():
    print('Load initial embedding vectors')
    init_word_embeds, entity_indices = ntn_input.load_init_embeds(params.data_path)
    print('Load input')
    pair_e = load_input()
    entity_indices = [entity_indices[pair_e[0]], entity_indices[pair_e[1]]]
    relation_list = ntn_input.load_relations(params.data_path)
    n_relations = len(relation_list)
    print('List of relations: {}'.format(relation_list))
    
    # define placeholder
    print('Define placeholder')
    data_plah = tf.placeholder(tf.int32, shape=(None, 2), name='pair_entities')
    
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
    
    # predict relation
    relation = predict_relation(data_plah[:, 0], data_plah[:, 1], entity_indices, n_relations, E, W, E, b, U)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('Load checkpoint ' + saved_model)
        saver = tf.train.Saver()
        saver.restore(sess)
        saver.restore(sess, params.output_path + saved_model + '.sess')
        print('Feed {} and {}'.format(pair_e[0], pair_e[1]))
        feed_dict = {data_plah[0]: pair_e[0], data_plah[1]: pair_e[1]}
        relation = sess.run([relation], feed_dict)
        print('{} and {} has {}'.format(pair_e[0], pair_e[1], relation_list[relation]))
