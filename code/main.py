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

word_vecs = [[4.0, 5.0, 6.0, 7.0],
             [8.0, 9.0, 10.0, 11.0],
             [12.0, 13.0, 14.0, 15.0],
             [16.0, 17.0, 18.0, 19.0],
             [20.0, 21.0, 22.0, 23.0],
             [24.0, 25.0, 26.0, 27.0]]

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
'''


def my_function(data_plah,
                indices,
                E, W, V, b, U):
    indices = [tf.constant(i) for i in indices]
    entity_vecs = tf.stack([tf.reduce_mean(tf.gather(E, i), 0) for i in indices])

    score_values = []
    for r in range(n_relations):
        e1, e2 = tf.split(data_plah[r], 2, axis=1)
        '''
        tf.gather results the shape of (?, 1, 4)    [[[18 19 20 21]][[12 13 14 15]]]
        tf.squeeze results the shape of (?, 4)      [[18 19 20 21][12 13 14 15]]
        tf.transpose results the shape of (4, ?) 
        '''
        e1 = tf.transpose(tf.squeeze(tf.gather(entity_vecs, e1, name='e1'), [1]))
        e2 = tf.transpose(tf.squeeze(tf.gather(entity_vecs, e2, name='e2'), [1]))

        # num_rel_r is the number of triples for each relation
        num_rel_r = tf.expand_dims(tf.shape(e1)[1], 0)

        term1 = []
        for i in range(slice_size):
            term1.append(tf.reduce_sum(e1 * tf.matmul(W[r][:, :, i], e2), 0))
        term1 = tf.stack(term1)

        term2 = tf.matmul(V[r], tf.concat([e1, e2], 0))

        tanh_function = tf.tanh(term1 + term2 + b[r])

        # TODO why need reshape
        score_value = tf.reshape(tf.matmul(U[r], tanh_function), num_rel_r)
        debug_0 = score_values
        foo = tf.Print(debug_0, [debug_0], message='score_value: ')

        score_values.append(score_value)

    return score_values, foo


def lab():
    indices = [[0, 1], [2], [4, 3], [0, 5, 1]]

    print('Define placeholders')
    data_plah = [tf.placeholder(tf.int32, shape=(None, 2), name='batch_' + str(i)) for i in range(n_relations)]
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

    score_values = my_function(data_plah, indices, E, W, V, b, U)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = {data_plah[0]: [[0, 1], [0, 3], [2, 1]], data_plah[1]: [[2, 3], [1, 2]]}

        score_values = sess.run(score_values, feed_dict)
        # print(score_values)


def main():
    # dataprocessing.generate_data(1)
    # if params.MODE == 1:
    #     evaluation.run_evaluation()
    lab()
    # production.run_production()


if __name__ == '__main__':
    main()
