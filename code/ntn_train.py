import tensorflow as tf
import ntn_input
import ntn
import params
import dataprocessing


def run_training():
    print('Begin training process')
    print('Load embedding word vector ...')
    init_word_embeds, entity_indices = ntn_input.load_init_embeds(params.data_path)

    print('Load training data from train.txt ...')
    train_data = ntn_input.load_training_data(params.data_path)

    print('Load entity list from entities.txt ...')
    entity_list = ntn_input.load_entities(params.data_path)

    # print('Load relation list from relations.txt ...')
    # relation_list = ntn_input.load_relations(params.data_path)

    print('Index raw data ...')
    train_data, fil_entity_indices, num_entities, num_relations = dataprocessing.index_data(train_data,
                                                                                            entity_list,
                                                                                            entity_indices)

    num_iters = params.num_iter
    batch_size = int(len(train_data) / 3)
    corrupt_size = params.corrupt_size
    slice_size = params.slice_size

    print('data_size: {} - batch_size: {} - corrupt_size: {} - num_relations: {}'.format(len(train_data), batch_size,
                                                                                         corrupt_size, num_relations))
    assert int(batch_size / num_relations) > corrupt_size

    with tf.Graph().as_default():
        sess = tf.Session()

        print('Create placeholders')
        placeholder_data = [tf.placeholder(tf.int32, shape=(None, 3), name='batch_' + str(i))
                            for i in range(num_relations)]
        placeholder_label = [tf.placeholder(tf.float32, shape=(None, 1), name='label_' + str(i))
                             for i in range(num_relations)]
        placeholder_corrupt = tf.placeholder(tf.bool, shape=(1))  # Which of e1 or e2 to corrupt?

        print('Create variables')
        E = tf.Variable(init_word_embeds)
        W = [tf.Variable(tf.truncated_normal([params.embedding_size, params.embedding_size, slice_size]))
             for _ in range(num_relations)]
        V = [tf.Variable(tf.zeros([slice_size, 2 * params.embedding_size]))
             for _ in range(num_relations)]
        b = [tf.Variable(tf.zeros([slice_size, 1]))
             for _ in range(num_relations)]
        U = [tf.Variable(tf.ones([1, slice_size]))
             for _ in range(num_relations)]

        print('Define model')
        inference = ntn.inference(placeholder_data,
                                  placeholder_corrupt,
                                  init_word_embeds,
                                  fil_entity_indices,
                                  num_entities,
                                  num_relations,
                                  slice_size,
                                  batch_size,
                                  False,
                                  placeholder_label,
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
            corrupting_batch = dataprocessing.generate_corrupting_batch(batch_size,
                                                                        train_data,
                                                                        num_entities,
                                                                        corrupt_size,
                                                                        num_relations)
            relation_batches = dataprocessing.split_corrupting_batch(corrupting_batch, num_relations)

            if i == params.save_per_iter:
                print('Save training model')
                saver.save(sess, params.output_path + "/" + params.data_name + str(i) + '.sess')

            print('Fill data')
            feed_dict = dataprocessing.fill_feed_dict(relation_batches,
                                                      params.train_both,
                                                      placeholder_data,
                                                      placeholder_label,
                                                      placeholder_corrupt)

            print('Execute computation graph')
            _, loss_value = sess.run([optimizer, loss], feed_dict=feed_dict)

        sess.close()


def main(argv):
    run_training()


if __name__ == "__main__":
    tf.app.run()
