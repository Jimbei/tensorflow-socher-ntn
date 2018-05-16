import tensorflow as tf
import ntn_input
import ntn
import params
import numpy as np
import numpy.matlib
import random
import datetime


def index_data(data, entities, relations):
    indexing_entities = {entities[i]: i for i in range(len(entities))}
    indexing_relations = {relations[i]: i for i in range(len(relations))}
    indexing_data = [(indexing_entities[data[i][0]],
                      indexing_relations[data[i][1]],
                      indexing_entities[data[i][2]])
                     for i in range(len(data))]

    return indexing_data


def get_batch(batch_size, data, num_entities, corrupt_size):
    random_indices = random.sample(range(len(data)), batch_size)
    batch = [(data[i][0],  # data[i][0] = e1
              data[i][1],  # data[i][1] = r
              data[i][2],  # data[i][2] = e2
              random.randint(0, num_entities - 1))  # random = e3 (corrupted)
             for i in random_indices for _ in range(corrupt_size)]

    return batch


def split_batch(data_batch, num_relations):
    batches = [[] for i in range(num_relations)]
    for e1, r, e2, e3 in data_batch:
        batches[r].append((e1, e2, e3))

    # print('sample batches[0][6]: {}'.format(batches[0][6]))
    # exit()

    return batches


def fill_feed_dict(batches, train_both, placeholder_data, placeholder_label, placeholder_corrupt):
    feed_dict = {placeholder_corrupt: [train_both and np.random.random() > 0.5]}

    for i in range(len(placeholder_data)):
        feed_dict[placeholder_data[i]] = batches[i]

        # print(batches[i])
        # exit()

        feed_dict[placeholder_label[i]] = [[0.0] for j in range(len(batches[i]))]

    return feed_dict


def filter_data(data, filtering_entity):
    filtering_data = []

    for i in range(len(data)):
        e1, r, e2 = data[i]
        if e1 in filtering_entity and e2 in filtering_entity:
            filtering_data.append(data[i])

    return filtering_data


def filter_entity(entity_indices):
    filtering_entity = []

    for i in entity_indices:
        for j in i:
            if j not in filtering_entity:
                filtering_entity.append(j)

    return filtering_entity


def run_training():
    print('Begin training process')
    print('Load training data from train.txt ...')
    train_data = ntn_input.load_training_data(params.data_path)

    print('Load entity list from entities.txt ...')
    entities_list = ntn_input.load_entities(params.data_path)

    print('Load relation list from relations.txt ...')
    relations_list = ntn_input.load_relations(params.data_path)

    print('Index raw data ...')
    train_data = index_data(train_data, entities_list, relations_list)
    # train_data = random.sample(train_data, 2000)

    print('Load embedding word vector ...')
    init_word_embeds, entity_indices = ntn_input.load_init_embeds(params.data_path)

    # # sample data
    # print('Sample data')
    # entity_indices = random.sample(entity_indices, 1000)
    # # filter entity_list
    # entity_indices = filter_entity(entity_indices)
    # # filter train_data
    # train_data = filter_data(train_data, entity_indices)
    # # filter relation_list
    # # =========================================================================

    num_entities = len(entities_list)
    num_relations = len(relations_list)

    num_iters = params.num_iter
    batch_size = params.batch_size
    corrupt_size = params.corrupt_size
    slice_size = params.slice_size

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
                                  entity_indices,
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
            data_batch = get_batch(batch_size,
                                   train_data,
                                   num_entities,  # 11
                                   corrupt_size)  # 10
            relation_batches = split_batch(data_batch, num_relations)

            # print('data_batch has type {} and size {}'.format(type(data_batch), np.array(data_batch).shape))
            # print('relation_batches has type {} and size {}'.format(type(relation_batches),
            #                                                         np.array(relation_batches).shape))
            #
            # exit()

            if i % params.save_per_iter == 0:
                print('Save training model')
                saver.save(sess, params.output_path + "/" + params.data_name + str(i) + '.sess')

            print('Fill data')
            feed_dict = fill_feed_dict(relation_batches,
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
