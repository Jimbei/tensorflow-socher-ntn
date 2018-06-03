import params
import evaluation
import dataprocessing
import tensorflow as tf
import production

'''
    There are two kinds of vector.
    1) Vector for a single word
    2) Vector for an entity
    
    An entity is made of several words such as: fucking_bitch
    
    
'''

slice_size = 1
embedding_size = 4
n_relations = 2
DEBUG_STR = '===== debug: '

word_vecs = [[4.0, 5.0, 6.0, 7.0],      # 0
             [8.0, 9.0, 10.0, 11.0],    # 1
             [12.0, 13.0, 14.0, 15.0],  # 2
             [16.0, 17.0, 18.0, 19.0],  # 3
             [20.0, 21.0, 22.0, 23.0],  # 4
             [24.0, 25.0, 26.0, 27.0]]  # 5

'''
indices =   [[0, 1], 
            [2], 
            [4, 3], 
            [0, 5, 1]]
            
triples =   [[0, 1],    0
            [0, 3],     0
            [2, 1],     0
            [2, 3],     1
            [1, 2]]     1
            
entity_vec =    [[6, 7, 8, 9],      0
                [12, 13, 14, 15],   1
                [18, 19, 20, 21],   2
                [12, 13, 14, 15]]   3
'''


def my_function(data_plah,
                indices,
                r,
                E, W, V, b, U):
    indices = [tf.constant(i) for i in indices]
    entity_vecs = tf.stack([tf.reduce_mean(tf.gather(E, i), 0) for i in indices])

    e1, e2 = tf.split(data_plah[r], 2, axis=1)
    '''
    tf.gather results the shape of (?, 1, 4)    [[[18 19 20 21]][[12 13 14 15]]]
    tf.squeeze results the shape of (?, 4)      [[18 19 20 21][12 13 14 15]]
    tf.transpose results the shape of (4, ?)
    The important point here is that "each ent_vec"
    '''
    # after transpose (4, 5)
    e1v = tf.transpose(tf.squeeze(tf.gather(entity_vecs, e1, name='e1'), [1]))
    e2v = tf.transpose(tf.squeeze(tf.gather(entity_vecs, e2, name='e2'), [1]))

    # num_rel_r is the number of triples for each relation
    num_rel_r = tf.expand_dims(tf.shape(e1v)[1], 0)

    term1 = []
    for i in range(slice_size):
        tmp1 = tf.matmul(e1v, W[r][:, :, i])
        tmp2 = tf.matmul(tmp1, e2v)
        tmp3 = tf.diag_part(tmp2)
        term1.append(tmp3)
    term1 = tf.stack(term1)

    term2 = tf.matmul(V[r], tf.concat([e1v, e2v], 0))

    tanh_function = tf.tanh(term1 + term2 + b[r])

    # TODO why need reshape
    score_value = tf.reshape(tf.matmul(U[r], tanh_function), num_rel_r)
    # debug
    foo = term1
    foo = tf.Print(foo, [tmp3], summarize=24, message='\n======== DEBUG - tmp3: ')
    # =====================================================================

    return score_value, foo


def lab():
    indices = [[0, 1], [2], [4, 3], [0, 5, 1]]

    print('Define placeholders')
    data_plah = [tf.placeholder(tf.int32, shape=(None, 2), name='batch_')]
    print('Define variables')
    E = tf.Variable(word_vecs)
    W = [tf.Variable(tf.ones([embedding_size, embedding_size, slice_size]))
         for _ in range(n_relations)]
    V = [tf.Variable(tf.ones([slice_size, 2 * embedding_size]))
         for _ in range(n_relations)]
    b = [tf.Variable(tf.ones([slice_size, 1]))
         for _ in range(n_relations)]
    U = [tf.Variable(tf.ones([1, slice_size]))
         for _ in range(n_relations)]

    score_r0 = my_function(data_plah, indices, 0, E, W, V, b, U)
    # score_r1 = my_function(data_plah, indices, 1, E, W, V, b, U)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # feed_dict = {data_plah[0]: [[0, 1], [0, 3], [2, 1]], data_plah[1]: [[2, 3], [1, 2]]}
        feed_dict = {data_plah[0]: [[0, 1], [0, 3], [2, 1], [2, 3], [1, 2]]}

        score_r0 = sess.run([score_r0], feed_dict)
        print(score_r0)


def main():
    # dataprocessing.generate_data(1)
    # if params.MODE == 1:
    #     evaluation.run_evaluation()
    lab()
    # production.run_production()


if __name__ == '__main__':
    main()
