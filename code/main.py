import params
import evaluation
import dataprocessing
import tensorflow as tf
import production

slice_size = 1
embedding_size = 4
n_relations = 2


def my_function(data_plah,
                indices,
                n_relations,
                slice_size,
                E, W, V, b, U):
    indices = [tf.constant(i) - 1 for i in indices]
    word_vecs = tf.stack([tf.reduce_mean(tf.gather(E, i), 0) for i in indices])

    predictions = []
    for r in range(n_relations):
        e1, e2 = tf.split(data_plah[r], 2, axis=1)
        e1 = tf.transpose(tf.squeeze(tf.gather(word_vecs, e1, name='e1'), [1]))
        e2 = tf.transpose(tf.squeeze(tf.gather(word_vecs, e2, name='e2'), [1]))
        
        num_rel_r = tf.expand_dims(tf.shape(e1)[1], 0)
        
        term1 = []
        for i in range(slice_size):
            term1.append(tf.reduce_sum(e1 * tf.matmul(W[r][:, :, i], e2), 0))
        term1 = tf.stack(term1)
        
        term2 = tf.matmul(V[r], tf.concat([e1, e2], 0))
        
        tanh_function = tf.tanh(term1 + term2 + b[r])
        
        score_value = tf.reshape(tf.matmul(U[r], tanh_function), num_rel_r)
        
        predictions.append(score_value)
        
    return predictions.index(max(predictions))


def lab():
    init_word_vecs = [[4, 5, 6, 7],
                      [8, 9, 10, 11],
                      [12, 13, 14, 15],
                      [16, 17, 18, 19],
                      [20, 21, 22, 23],
                      [24, 25, 26, 27]]
    indices = [[0, 1], [2], [4, 3], [0, 5, 1]]
    
    feed_dict = {}
    
    print('Define placeholders')
    data_plah = [tf.placeholder(tf.int32, shape=(None, 3), name='batch_' + str(i)) for i in range(n_relations)]
    print('Define variables')
    E = tf.Variable(init_word_vecs)
    W = [tf.Variable(tf.ones([embedding_size, embedding_size, slice_size]))
         for _ in range(n_relations)]
    V = [tf.Variable(tf.ones([slice_size, 2 * embedding_size]))
         for _ in range(n_relations)]
    b = [tf.Variable(tf.ones([slice_size, 1]))
         for _ in range(n_relations)]
    U = [tf.Variable(tf.ones([1, slice_size]))
         for _ in range(n_relations)]
    
    my_function(data_plah, indices, n_relations, slice_size, E, W, V, b, U)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())


def main():
    # dataprocessing.generate_data(1)
    # if params.MODE == 1:
    #     evaluation.run_evaluation()
    lab()
    # production.run_production()


if __name__ == '__main__':
    main()
