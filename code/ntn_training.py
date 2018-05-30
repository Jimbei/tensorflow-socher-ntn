import tensorflow as tf
import ntn_input
import ntn
import params
import datamanipulation as dp


def run_training():
    print('Begin training process')
    print('Load embedding word vector ...')
    init_word_vecs, entity_indices = ntn_input.load_init_embeds(params.DATA_DIR)
    
    print('Load training data from train.txt ...')
    triples = ntn_input.load_triples(params.DATA_DIR)
    
    print('Load entity list from entities.txt ...')
    entity_list = ntn_input.load_entities(params.DATA_DIR)
    
    # print('Load relation list from relations.txt ...')
    # relation_list = ntn_input.load_relations(params.DATA_DIR)
    
    print('Index raw data ...')
    triples, fil_entity_indices, n_entities, n_relations = dp.index_data(triples,
                                                                         entity_list,
                                                                         entity_indices)
    
    num_iters = params.n_iters
    batch_size = int(len(triples) / 3)
    corrupt_size = params.corrupt_size
    slice_size = params.slice_size
    
    print('data_size: {} - batch_size: {} - corrupt_size: {} - num_relations: {}'.format(len(triples), batch_size,
                                                                                         corrupt_size, n_relations))
    assert int(batch_size / n_relations) > corrupt_size
    
    with tf.Graph().as_default():
        sess = tf.Session()
        
        print('Create placeholders')
        features = [tf.placeholder(tf.int32, shape=(None, 3), name='batch_' + str(i))
                    for i in range(n_relations)]
        labels = [tf.placeholder(tf.float32, shape=(None, 1), name='label_' + str(i))
                  for i in range(n_relations)]
        corrupting = tf.placeholder(tf.bool, shape=(1))  # Which of e1 or e2 to corrupt?
        
        print('Create variables')
        # TODO E.shape == (38696, 100)
        E = tf.Variable(init_word_vecs)
        W = [tf.Variable(tf.truncated_normal([params.embedding_size, params.embedding_size, slice_size]))
             for _ in range(n_relations)]
        V = [tf.Variable(tf.zeros([slice_size, 2 * params.embedding_size]))
             for _ in range(n_relations)]
        b = [tf.Variable(tf.zeros([slice_size, 1]))
             for _ in range(n_relations)]
        U = [tf.Variable(tf.ones([1, slice_size]))
             for _ in range(n_relations)]
        
        print('Define hypothesis function')
        inference = ntn.inference(features,
                                  corrupting,
                                  init_word_vecs,
                                  fil_entity_indices,
                                  n_entities,
                                  n_relations,
                                  slice_size,
                                  batch_size,
                                  False,
                                  labels,
                                  E, W, V, b, U)
        
        print('Define loss function')
        loss = ntn.loss(inference, params.regularization)
        
        print('Define optimizer function')
        optimizer = ntn.training(loss, params.learning_rate)
        
        print('Initialize saver')
        saver = tf.train.Saver(tf.trainable_variables())
        
        print('Initialize variables')
        sess.run(tf.global_variables_initializer())
        
        for i in range(1, num_iters):
            print('#iter: {}'.format(i))
            # TODO combine generate_corrupting_batch and split_corrupting_batch
            corrupting_batch = dp.generate_corrupting_batch(batch_size,
                                                            triples,
                                                            n_entities,
                                                            corrupt_size,
                                                            n_relations)
            relation_batches = dp.split_corrupting_batch(corrupting_batch, n_relations)
            
            if i == params.save_per_iter:
                print('Save training model')
                saver.save(sess, params.output_path + "/" + params.data_name + str(i) + '.sess')
            
            print('Fill data')
            feed_dict = dp.fill_feed_dict(relation_batches,
                                          params.train_both,
                                          features,
                                          labels,
                                          corrupting)
            
            print('Execute computation graph')
            _, loss_value = sess.run([optimizer, loss], feed_dict=feed_dict)
        
        sess.close()


def main(argv):
    run_training()


if __name__ == "__main__":
    tf.app.run()
