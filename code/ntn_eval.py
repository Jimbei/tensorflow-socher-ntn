import datman
import hypothesis
import params
import tensorflow as tf
import numpy as np
import evaluation
import random

saved_model = 'Wordnet490'
CKPT_DIR = params.output_path + saved_model + '.sess'


def index_data(data, entity_list, relations):
    entity_to_index = {entity_list[i]: i for i in range(len(entity_list))}
    relation_to_index = {relations[i]: i for i in range(len(relations))}
    indexed_data = [(entity_to_index[data[i][0]],
                     relation_to_index[data[i][1]],
                     entity_to_index[data[i][2]],
                     float(data[i][3]))
                    for i in range(len(data))]

    return indexed_data


def fill_feed_dict(batches, labels, train_both, batch_placeholders, label_placeholders, corrupt_placeholder):
    feed_dict = {corrupt_placeholder: [train_both and np.random.random() > 0.5]}
    for i in range(len(batch_placeholders)):
        feed_dict[batch_placeholders[i]] = batches[i]
    for i in range(len(label_placeholders)):
        feed_dict[label_placeholders[i]] = labels[i]
    return feed_dict


# dataset is in the form (e1, R, e2, label)
def data_to_relation_sets(data_batch, num_relations):
    batches = [[] for _ in range(num_relations)]
    labels = [[] for _ in range(num_relations)]

    for e1, r, e2, label in data_batch:
        batches[r].append((e1, e2, 1))
        labels[r].append([label])

    return batches, labels


def run_evaluation():
    print('Begin evaluation process')
    print('Load data from test.txt ...')
    test_data = datman.load_test_data(params.data_path)

    print('Load entity list from entities.txt ...')
    entities_list = datman.load_entities(params.data_path)

    print('Load relation list from relations.txt ...')
    relations_list = datman.load_relations(params.data_path)

    print('Index raw data ...')
    test_data = index_data(test_data, entities_list, relations_list)

    batch_size = len(test_data)
    num_entities = len(entities_list)
    num_relations = len(relations_list)

    slice_size = params.slice_size
    init_word_embeds, entity_to_wordvec = datman.load_init_embeds(params.data_path)
    batches, labels = data_to_relation_sets(test_data, num_relations)

    with tf.Graph().as_default():
        sess = tf.Session()

        print('Create placeholders')
        batch_placeholders = [tf.placeholder(tf.float32, shape=(None, 3)) for i in range(num_relations)]
        label_placeholders = [tf.placeholder(tf.float32, shape=(None, 1)) for i in range(num_relations)]
        corrupt_placeholder = tf.placeholder(tf.bool, shape=(1))

        E = tf.Variable(init_word_embeds)
        W = [tf.Variable(tf.truncated_normal([params.embedding_size, params.embedding_size, slice_size]))
             for _ in range(num_relations)]
        V = [tf.Variable(tf.zeros([slice_size, 2 * params.embedding_size]))
             for _ in range(num_relations)]
        b = [tf.Variable(tf.zeros([slice_size, 1]))
             for _ in range(num_relations)]
        U = [tf.Variable(tf.ones([1, slice_size]))
             for _ in range(num_relations)]

        print('Define hypothesis function')
        inference = hypothesis.hypothesis(batch_placeholders,
                                          corrupt_placeholder,
                                          init_word_embeds,
                                          entity_to_wordvec,
                                          num_entities,
                                          num_relations,
                                          slice_size,
                                          batch_size,
                                          True,
                                          label_placeholders,
                                          E, W, V, b, U)
        eval_correct = hypothesis.eval(inference)

        assert CKPT_DIR == '../output/Wordnet/Wordnet490.sess'
        print('Load checkpoint {}'.format(saved_model))
        saver = tf.train.Saver()
        saver.restore(sess, params.output_path + saved_model + '.sess')

        # print('Initialize variable')
        # sess.run(tf.global_variables_initializer())

        do_eval(sess,
                eval_correct,
                batch_placeholders,
                label_placeholders,
                corrupt_placeholder,
                batches,
                labels,
                batch_size)


def do_eval(sess,
            eval_correct,
            batch_placeholders,
            label_placeholders,
            corrupt_placeholder,
            test_batches,
            test_labels,
            n_samples):
    true_count = 0.

    feed_dict = fill_feed_dict(test_batches,
                               test_labels,
                               params.train_both,
                               batch_placeholders,
                               label_placeholders,
                               corrupt_placeholder)

    score_values, labels, debug_printout = sess.run(eval_correct, feed_dict)

    for i in range(len(score_values[0])):
        if score_values[0][i] > 0 and labels[0][i] == 1:
            true_count += 1.0
        elif score_values[0][i] < 0 and labels[0][i] == -1:
            true_count += 1.0
    precision = float(true_count) / float(n_samples)

    print('precision: {}'.format(precision))
    sess.close()
    return precision


def get_thresholds(batch_size):
    dev_data = datman.load_dev_data()
    entities_list = datman.load_entities(params.data_path)
    relations_list = datman.load_entities(params.data_path)

    n_entities = len(entities_list)
    n_relations = len(relations_list)

    slice_size = params.slice_size
    init_word_embeds, entity_indices = datman.load_init_embeds(params.data_path)

    data_plah = tf.placeholder(tf.float32, shape=(4, batch_size))
    corrupt_plah = tf.placeholder(tf.bool, shape=(1))  # Which of e1 or e2 to corrupt?
    score_values = hypothesis.hypothesis(data_plah, corrupt_plah, init_word_embeds, entity_indices,
                                         n_entities, n_relations, slice_size, batch_size)

    min_score = tf.reduce_min(score_values)
    max_score = tf.reduce_max(score_values)

    # initialize thresholds and accuracies
    best_thresholds = tf.zeros([n_relations, 1])
    best_accuracies = tf.zeros([n_relations, 1])

    for r in range(n_relations):
        best_thresholds[r, :] = min_score
        best_accuracies[r, :] = -1

    score = min_score
    increment = 0.01

    while score <= max_score:
        # iterate through relations list to find 
        for r in range(n_relations):
            current_relation_list = (dev_data[:, 1] == r)
            dev_labels = dev_data[:, 3]
            predictions = (score_values[current_relation_list, 0] <= score) * 2 - 1
            accuracy = tf.reduce_mean((predictions == dev_labels[current_relation_list, 0]))

            # update threshold and accuracy
            if accuracy > best_accuracies[r, 0]:
                best_accuracies[r, 0] = accuracy
                best_thresholds[r, 0] = score

        score += increment

    # store threshold values
    return best_thresholds


def get_predictions(batch_size):
    best_thresholds = get_thresholds()

    triples = datman.load_test_data()
    entity_list = datman.load_entities(params.data_path)
    relation_list = datman.load_entities(params.data_path)

    n_entities = len(entity_list)
    n_relations = len(relation_list)

    slice_size = params.slice_size
    (init_word_embeds, entity_to_wordvec) = datman.load_init_embeds(params.data_path)

    batch_placeholder = tf.placeholder(tf.float32, shape=(4, batch_size))
    corrupt_placeholder = tf.placeholder(tf.bool, shape=(1))  # Which of e1 or e2 to corrupt?
    score_values = hypothesis.hypothesis(batch_placeholder, corrupt_placeholder, init_word_embeds,
                                         entity_to_wordvec,
                                         n_entities, n_relations, slice_size, batch_size)

    predictions = tf.zeros((triples.shape[0], 1))
    for i in range(triples.shape[0]):
        # get relation
        rel = triples[i, 1]

        # get labels based on predictions
        if score_values[i, 0] <= best_thresholds[rel, 0]:
            predictions[i, 0] = 1
        else:
            predictions[i, 0] = -1

    return predictions


if __name__ == "__main__":
    evaluation.run_evaluation()
    # run_evaluation()
