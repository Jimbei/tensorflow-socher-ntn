import tensorflow as tf
import ntn_input  # library for reading data
import ntn
import params
import numpy as np
import numpy.matlib
import random
import datetime


def index_training_data(raw_training_data, entities_list, relations_list):
    """
    Encode training data into number

    :param raw_training_data:
    :param entities_list:
    :param relations_list:
    :return: indexing training data
    """
    entity_to_index = {entities_list[i]: i for i in range(len(entities_list))}
    relation_to_index = {relations_list[i]: i for i in range(len(relations_list))}
    indexing_training_data = [(entity_to_index[raw_training_data[i][0]],
                               relation_to_index[raw_training_data[i][1]],
                               entity_to_index[raw_training_data[i][2]])
                              for i in range(len(raw_training_data))]
    return indexing_training_data


def generate_corrupting_data(corrupting_data_size, indexed_training_data, num_entities, corrupting_sample_size):
    """
    Create randomly a sublist of indexed_training_data with the size of corrupting_data_size

    :param corrupting_data_size: 20000
    :param indexed_training_data:
    :param num_entities:
    :param corrupting_sample_size: 10
    :return: corrupting_data
    """

    # generate a `list of random indices` of training data
    random_indices = random.sample(range(len(indexed_training_data)), corrupting_data_size)

    # fill in the list
    indexed_corrupting_data = [(indexed_training_data[i][0],
                                indexed_training_data[i][1],
                                indexed_training_data[i][2],
                                random.randint(0, num_entities - 1))  # corrupting sample chosen randomly
                               for i in random_indices for j in range(corrupting_sample_size)]
    return indexed_corrupting_data


def tidy_corrupting_data(indexing_corrupting_data, num_relations):
    """
    Tidy corrupting training data

    :param indexing_corrupting_data: original corrupting data
    :param num_relations: number of relations used in dataset
    :return: tidy corrupting training data
    """
    relation_corrupting_data = [[] for i in range(num_relations)]
    for e1, r, e2, e3 in indexing_corrupting_data:
        relation_corrupting_data[r].append((e1, e2, e3))
    return relation_corrupting_data


def fill_feed_dict(relation_batches, train_both, batch_placeholders, label_placeholders, corrupt_placeholder):
    feed_dict = {corrupt_placeholder: [train_both and np.random.random() > 0.5]}
    for i in range(len(batch_placeholders)):
        feed_dict[batch_placeholders[i]] = relation_batches[i]
        feed_dict[label_placeholders[i]] = [[0.0] for j in range(len(relation_batches[i]))]
    return feed_dict


def run_training():
    print("Begin!")
    # python list of (e1, R, e2) for entire training set in string form
    print("Load training data from train.txt...")
    raw_training_data = ntn_input.load_training_data(params.data_path)
    print("Load entities from entities.txt ...")
    entities_list = ntn_input.load_entities(params.data_path)
    print('Load relations from relations.txt...')
    relations_list = ntn_input.load_relations(params.data_path)
    # python list of (e1, R, e2) for entire training set in index form
    print('Indexing training data ...')
    indexing_training_data = index_training_data(raw_training_data, entities_list, relations_list)
    print("Load initial embedding parameters ...")
    word_vecs, entity_indices = ntn_input.load_init_embeds(params.data_path)

    num_entities = len(entities_list)
    num_relations = len(relations_list)

    num_iters = params.num_iter
    corrupting_data_size = params.batch_size  # 20000
    corrupt_size = params.corrupt_size  # 10
    slice_size = params.slice_size  # 3

    with tf.Graph().as_default():
        print("Starting to build graph " + str(datetime.datetime.now()))
        batch_placeholders = [tf.placeholder(tf.int32, shape=(None, 3), name='batch_' + str(i)) for i in
                              range(num_relations)]
        label_placeholders = [tf.placeholder(tf.float32, shape=(None, 1), name='label_' + str(i)) for i in
                              range(num_relations)]

        corrupt_placeholder = tf.placeholder(tf.bool, shape=(1))  # Which of e1 or e2 to corrupt?

        # ====  Build Model ====
        prediction_values = ntn.g_function(batch_placeholders,
                                           corrupt_placeholder,
                                           word_vecs,
                                           entity_indices,
                                           num_entities,
                                           num_relations,
                                           slice_size,
                                           corrupting_data_size,
                                           False,
                                           label_placeholders)
        # =====================================================================

        # ==== Define loss function ====
        loss = ntn.loss(prediction_values, params.regularization)
        # =====================================================================

        # ==== Define training algorithm ====
        training = ntn.training(loss, params.learning_rate)
        # =====================================================================

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Run the Op to initialize the variables.
        init = tf.initialize_all_variables()
        sess.run(init)
        saver = tf.train.Saver(tf.trainable_variables())
        for i in range(1, num_iters):
            print("Starting iter " + str(i) + " " + str(datetime.datetime.now()))
            indexing_corrupting_data = generate_corrupting_data(corrupting_data_size,
                                                                indexing_training_data,
                                                                num_entities,
                                                                corrupt_size)
            relation_corrupting_data = tidy_corrupting_data(indexing_corrupting_data, num_relations)

            if i % params.save_per_iter == 0:
                saver.save(sess, params.output_path + "/" + params.data_name + str(i) + '.sess')

            feed_dict = fill_feed_dict(relation_corrupting_data,
                                       params.train_both,
                                       batch_placeholders,
                                       label_placeholders,
                                       corrupt_placeholder)
            _, loss_value = sess.run([training, loss], feed_dict=feed_dict)

            # TODO: Eval against dev set?


def main(argv):
    run_training()


if __name__ == "__main__":
    print("Welcome to Neural Tensor Network")
    tf.app.run()
