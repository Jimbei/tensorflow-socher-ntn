import tensorflow as tf
import numpy as np
import random

import ntn_input
import params
import hypothesis
import datamanipulation as dp


def index_data(data, entity_list, entity_indices):
    print('Sample data')
    fil_entity_list = random.sample(entity_list, 2000)
    data = dp.filter_data(data, fil_entity_list, 1)
    # print('original relation list: {}'.format(relation_list))
    fil_relation_list = dp.filter_relation(data, 1)
    fil_entity_indices = dp.filter_entity_indices(entity_list, entity_indices, fil_entity_list)

    indexing_entities = {fil_entity_list[i]: i for i in range(len(fil_entity_list))}
    indexing_relations = {fil_relation_list[i]: i for i in range(len(fil_relation_list))}
    indexing_data = [(indexing_entities[data[i][0]],
                      indexing_relations[data[i][1]],
                      indexing_entities[data[i][2]],
                      float(data[i][3])) for i in range(len(data))]

    n_entities = len(fil_entity_list)
    n_relations = len(fil_relation_list)

    return indexing_data, fil_entity_indices, n_entities, n_relations


def fill_feed_dict(data_plah, label_plah, corrupt_plah, batches, labels):
    feed_dict = {corrupt_plah: [False and np.random.random() > 0.5]}

    for i in range(len(data_plah)):
        feed_dict[data_plah[i]] = batches[i]
    for i in range(len(label_plah)):
        print(i)
        feed_dict[label_plah[i]] = [labels[i]]

    return feed_dict


# dataset is in the form (e1, R, e2, label)
def data_to_relation_sets(data_batch, n_relations):
    batches = [[] for _ in range(n_relations)]
    labels = [[] for _ in range(n_relations)]
    for e1, r, e2, label in data_batch:
        batches[r].append([e1, e2, 1])
        labels[r].append([label])
    return batches, labels


def run_evaluation():
    print('Load initial embedding word vectors')
    init_word_vecs, entity_indices = dp.load_init_embeds(params.DATA_DIR)
    print('Load evaluation data')
    triples = dp.load_triples(params.DATA_DIR, 1)
    entities_list = dp.load_entities(params.DATA_DIR)
    print('Index data')
    triples, entity_indices, n_entities, n_relations = index_data(triples, entities_list, entity_indices)

    batch_size = len(triples)
    batches, labels = data_to_relation_sets(triples, n_relations)

    print('Define placeholders')
    data_plah = [tf.placeholder(tf.int32, shape=(None, 3), name='batch_' + str(i))
                 for i in range(n_relations)]
    label_plah = [tf.placeholder(tf.float32, shape=(None, 1), name='label_' + str(i))
                  for i in range(n_relations)]
    corrupt_plah = tf.placeholder(tf.bool, shape=(1))

    print('Define variables')
    E = tf.Variable(init_word_vecs)
    W = [tf.Variable(tf.truncated_normal([params.embedding_size, params.embedding_size, params.slice_size]))
         for _ in range(n_relations)]
    V = [tf.Variable(tf.zeros([params.slice_size, 2 * params.embedding_size]))
         for _ in range(n_relations)]
    b = [tf.Variable(tf.zeros([params.slice_size, 1]))
         for _ in range(n_relations)]
    U = [tf.Variable(tf.ones([1, params.slice_size]))
         for _ in range(n_relations)]

    print('Define hypothesis')
    score_values = hypothesis.hypothesis(data_plah,
                                         label_plah,
                                         entity_indices,
                                         n_relations,
                                         E, W, V, b, U,
                                         True)
    score_values, labels = tf.split(score_values, 2, axis=0)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        print('Restore checkpoint')
        saver.restore(sess, params.output_path + 'Wordnet70.sess')
        print('Feed data')
        feed_dict = fill_feed_dict(data_plah,
                                   label_plah,
                                   corrupt_plah,
                                   batches,
                                   labels)
        score_values, labels = sess.run(score_values, feed_dict)

        print('Get {} accuracy on dataset {}'.format(get_precision(score_values,
                                                                   labels,
                                                                   batch_size), 'Wordnet'))


def get_precision(score_values, labels, batch_size):
    true_count = 0.
    for i in range(len(score_values[0])):
        if score_values[0][i] > 0 and labels[0][i] == 1:
            true_count += 1.0
        elif score_values[0][i] < 0 and labels[0][i] == -1:
            true_count += 1.0
    precision = float(true_count) / float(batch_size)
    return precision


def get_thresholds():
    dev_data = ntn_input.load_dev_data()
    entities_list = ntn_input.load_entities(params.DATA_DIR)
    relations_list = ntn_input.load_entities(params.DATA_DIR)

    num_entities = len(entities_list)
    num_relations = len(relations_list)

    slice_size = params.slice_size
    (init_word_embeds, entity_to_wordvec) = ntn_input.load_init_embeds(params.DATA_DIR)

    batch_placeholder = tf.placeholder(tf.float32, shape=(4, batch_size))
    corrupt_placeholder = tf.placeholder(tf.bool, shape=(1))  # Which of e1 or e2 to corrupt?
    predictions_list = evaluation.inference(batch_placeholder, corrupt_placeholder, init_word_embeds, entity_to_wordvec,
                                            num_entities, num_relations, slice_size, batch_size)

    min_score = tf.reduce_min(predictions_list)
    max_score = tf.reduce_max(predictions_list)

    # initialize thresholds and accuracies
    best_thresholds = tf.zeros([params.num_relations, 1])
    best_accuracies = tf.zeros([params.num_relations, 1])

    for i in range(params.num_relations):
        best_thresholds[i, :] = score_min
        best_accuracies[i, :] = -1

    score = min_score
    increment = 0.01

    while (score <= max_score):
        # iterate through relations list to find
        for i in range(params.num_relations):
            current_relation_list = (dev_data[:, 1] == i)
            predictions = (predictions_list[current_relation_list, 0] <= score) * 2 - 1
            accuracy = tf.reduce_mean((predictions == dev_labels[current_relations_list, 0]))

            # update threshold and accuracy
            if (accuracy > best_accuracies[i, 0]):
                best_accuracies[i, 0] = accuracy
                best_thresholds[i, 0] = score

        score += increment

    # store threshold values
    return best_thresholds


def get_predictions():
    best_thresholds = get_thresholds()
    test_data = ntn_input.load_test_data()
    entities_list = ntn_input.load_entities(params.DATA_DIR)
    relations_list = ntn_input.load_entities(params.DATA_DIR)

    num_entities = len(entities_list)
    num_relations = len(relations_list)

    slice_size = params.slice_size
    (init_word_embeds, entity_to_wordvec) = ntn_input.load_init_embeds(params.DATA_DIR)

    batch_placeholder = tf.placeholder(tf.float32, shape=(4, batch_size))
    corrupt_placeholder = tf.placeholder(tf.bool, shape=(1))  # Which of e1 or e2 to corrupt?
    predictions_list = evaluation.inference(batch_placeholder, corrupt_placeholder, init_word_embeds, entity_to_wordvec,
                                            num_entities, num_relations, slice_size, batch_size)

    predictions = tf.zeros((test_data.shape[0], 1))
    for i in range(test_data.shape[0]):
        # get relation
        rel = test_data[i, 1]

        # get labels based on predictions
        if (preictions_list[i, 0] <= self.best_thresholds[rel, 0]):
            predictions[i, 0] = 1
        else:
            predictions[i, 0] = -1

    return predictions


if __name__ == "__main__":
    run_evaluation()
