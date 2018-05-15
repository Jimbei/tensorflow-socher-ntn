import tensorflow as tf
import ntn_input
import ntn
import params
import numpy as np
import numpy.matlib
import random
import datetime


def index_data(data, entities, relations):
    entity_to_index = {entities[i]: i for i in range(len(entities))}
    relation_to_index = {relations[i]: i for i in range(len(relations))}
    indexing_data = [(entity_to_index[data[i][0]],
                      relation_to_index[data[i][1]],
                      entity_to_index[data[i][2]])
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
    return batches


def fill_feed_dict(batches, train_both, batch_placeholders, label_placeholders, corrupt_placeholder):
    feed_dict = {corrupt_placeholder: [train_both and np.random.random() > 0.5]}
    for i in range(len(batch_placeholders)):
        feed_dict[batch_placeholders[i]] = batches[i]
        feed_dict[label_placeholders[i]] = [[0.0] for j in range(len(batches[i]))]
    return feed_dict


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
    train_data = random.sample(train_data, 1000)

    print('Load embedding word vector ...')
    init_word_embeds, entity_to_wordvec = ntn_input.load_init_embeds(params.data_path)

    num_entities = len(entities_list)
    num_relations = len(relations_list)

    num_iters = params.num_iter
    batch_size = params.batch_size
    corrupt_size = params.corrupt_size
    slice_size = params.slice_size

    with tf.Graph().as_default():
        print('Create placeholders')
        data_placeholders = [tf.placeholder(tf.int32, shape=(None, 3), name='batch_' + str(i))
                             for i in range(num_relations)]
        label_placeholders = [tf.placeholder(tf.float32, shape=(None, 1), name='label_' + str(i))
                              for i in range(num_relations)]

        corrupt_placeholder = tf.placeholder(tf.bool, shape=(1))  # Which of e1 or e2 to corrupt?

        print('Define training model')
        inference = ntn.inference(data_placeholders,
                                  corrupt_placeholder,
                                  init_word_embeds,
                                  entity_to_wordvec,
                                  num_entities,
                                  num_relations,
                                  slice_size,
                                  batch_size,
                                  False,
                                  label_placeholders)

        print('Define loss function')
        loss = ntn.loss(inference, params.regularization)

        print('Define training algorithm')
        training = ntn.training(loss, params.learning_rate)

        # Create a session for running Ops on the Graph.
        sess = tf.Session()
        # print('Initialize variables')
        # sess.run(tf.global_variables_initializer())

        print('Initialize saver')
        saver = tf.train.Saver(tf.trainable_variables())

        for i in range(1, num_iters):
            print('#iter: {}'.format(i))
            data_batch = get_batch(batch_size, train_data, num_entities, corrupt_size)
            relation_batches = split_batch(data_batch, num_relations)

            if i % params.save_per_iter == 0:
                print('Save training model')
                saver.save(sess, params.output_path + "/" + params.data_name + str(i) + '.sess')

            print('Fill data')
            feed_dict = fill_feed_dict(relation_batches,
                                       params.train_both,
                                       data_placeholders,
                                       label_placeholders,
                                       corrupt_placeholder)

            print('Execute computation graph')
            _, loss_value = sess.run([training, loss], feed_dict=feed_dict)

        sess.close()


def main(argv):
    run_training()


if __name__ == "__main__":
    tf.app.run()
