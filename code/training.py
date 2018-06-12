import tensorflow as tf
import datman
import hypothesis
import params


def run_training():
    print('Begin training process')
    print('Load embedding word vector ...')
    init_word_embeds, entity_indices = datman.load_init_embeds(params.data_path)

    print('Load training data from train.txt ...')
    data = datman.load_training_data(params.data_path)

    print('Load entity list from entities.txt ...')
    entity_list = datman.load_entities(params.data_path)

    # print('Load relation list from relations.txt ...')
    # relation_list = ntn_input.load_relations(params.data_path)

    print('Index raw data ...')
    data, fil_entity_indices, num_entities, n_relations = datman.index_data(data,
                                                                            entity_list,
                                                                            entity_indices)

    num_iters = params.num_iter
    batch_size = int(len(data) / 3)

    print('======== DEBUG: \n'
          '\tdata_size: {}\n'
          '\tbatch_size: {}\n'
          '\tcorrupt_size: {}\n'
          '\tn_relations: {}'.format(len(data), batch_size, params.corrupt_size, n_relations))

    with tf.Graph().as_default():
        sess = tf.Session()

        print('Create placeholders')
        data_plah = [tf.placeholder(tf.int32, shape=(None, 3), name='batch_' + str(i))
                     for i in range(n_relations)]
        placeholder_label = [tf.placeholder(tf.float32, shape=(None, 1), name='label_' + str(i))
                             for i in range(n_relations)]
        placeholder_corrupt = tf.placeholder(tf.bool, shape=(1))

        print('Create variables')
        # E.shape == (38696, 100)
        E = tf.Variable(init_word_embeds)
        W = [tf.Variable(tf.truncated_normal([params.embedding_size, params.embedding_size, params.slice_size]))
             for _ in range(n_relations)]
        V = [tf.Variable(tf.zeros([params.slice_size, 2 * params.embedding_size]))
             for _ in range(n_relations)]
        b = [tf.Variable(tf.zeros([params.slice_size, 1]))
             for _ in range(n_relations)]
        U = [tf.Variable(tf.ones([1, params.slice_size]))
             for _ in range(n_relations)]

        print('Define hypothesis function')
        score_values, foo = hypothesis.hypothesis(data_plah,
                                                  fil_entity_indices,
                                                  n_relations,
                                                  E, W, V, b, U)

        print('Define loss function')
        loss = hypothesis.loss(score_values, params.regularization)

        print('Define optimizer function')
        optimizer = hypothesis.training(loss, params.learning_rate)

        print('Initialize saver')
        saver = tf.train.Saver(tf.trainable_variables())

        print('Initialize variables')
        sess.run(tf.global_variables_initializer())

        for i in range(1, num_iters):
            print('#iter: {}'.format(i))
            corrupting_batch = datman.generate_corrupting_batch(batch_size,
                                                                data,
                                                                num_entities,
                                                                n_relations)
            relation_batches = datman.split_corrupting_batch(corrupting_batch, n_relations)

            if i == params.save_per_iter:
                print('Save training model')
                saver.save(sess, params.output_path + "/" + params.data_name + str(i) + '.sess')

            print('Fill data')
            feed_dict = datman.fill_feed_dict(relation_batches,
                                              params.train_both,
                                              data_plah,
                                              placeholder_label,
                                              placeholder_corrupt)

            print('Execute computation graph')
            _, foo_value, loss_value = sess.run([optimizer, foo, loss], feed_dict=feed_dict)

        sess.close()


def main():
    run_training()


if __name__ == "__main__":
    main()
