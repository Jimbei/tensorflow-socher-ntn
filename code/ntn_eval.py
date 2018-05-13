import numpy as np
import tensorflow as tf
# defined lib
import ntn_input
import ntn
import params


# TODO: where to load model
# TODO: how to load a pair of entities (e1, e2), then get the corresponding R

def index_data(raw_data, entity_list, relation_list):
    """
    Index raw data into number

    :param raw_data:
    :param entity_list:
    :param relation_list:
    :return:
    """
    entity_to_index = {entity_list[i]: i for i in range(len(entity_list))}
    relation_to_index = {relation_list[i]: i for i in range(len(relation_list))}
    indexing_data = [(entity_to_index[raw_data[i][0]],
                      relation_to_index[raw_data[i][1]],
                      entity_to_index[raw_data[i][2]],
                      float(raw_data[i][3]))  # the value of each triple [-1; 1]
                     for i in range(len(raw_data))]
    return indexing_data


def fill_feed_dict(batches, labels, train_both, batch_placeholders, label_placeholders, corrupt_placeholder):
    feed_dict = {corrupt_placeholder: [train_both and np.random.random() > 0.5]}

    for i in range(len(batch_placeholders)):
        feed_dict[batch_placeholders[i]] = batches[i]

    for i in range(len(label_placeholders)):
        feed_dict[label_placeholders[i]] = labels[i]

    return feed_dict


# dataset is in the form (e1, R, e2, label)
def data_to_relation_sets(data_batch, num_relations):
    batches = [[] for i in range(num_relations)]
    labels = [[] for i in range(num_relations)]
    for e1, r, e2, label in data_batch:
        batches[r].append((e1, e2, 1))
        labels[r].append([label])
    return (batches, labels)


def run_evaluation():
    print(params.output_path)
    print(tf.train.latest_checkpoint(params.output_path, 'checkpoint'))

    print('Load entities from entities.txt ...')
    entities_list = ntn_input.load_entities(params.data_path)

    print('Load relations from relations.txt ...')
    relations_list = ntn_input.load_relations(params.data_path)

    print('Load raw data from test.txt ...')
    test_data = ntn_input.load_test_data(params.data_path)

    print('Index raw data ...')
    test_data = index_data(test_data, entities_list, relations_list)

    batch_size = len(test_data)

    num_entities = len(entities_list)
    num_relations = len(relations_list)

    slice_size = params.slice_size
    (init_word_embeds, entity_to_wordvec) = ntn_input.load_init_embeds(params.data_path)
    batches, labels = data_to_relation_sets(test_data, num_relations)

    with tf.Graph().as_default():
        sess = tf.Session()
        batch_placeholders = [tf.placeholder(tf.float32, shape=(None, 3)) for i in range(num_relations)]
        label_placeholders = [tf.placeholder(tf.float32, shape=(None, 1)) for i in range(num_relations)]
        corrupt_placeholder = tf.placeholder(tf.bool, shape=(1))

        inference = ntn.g_function(batch_placeholders,
                                   corrupt_placeholder,
                                   init_word_embeds,
                                   entity_to_wordvec,
                                   num_entities,
                                   num_relations,
                                   slice_size,
                                   batch_size,
                                   True,
                                   label_placeholders)

        eval_correct = ntn.eval(inference)
        saver = tf.train.Saver()

        saver.restore(sess, params.output_path + 'around100/Wordnet70.sess')
        # init = tf.initialize_all_variables()
        # sess.run(init)
        print('Begin evaluation')
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
            num_examples):
    true_count = 0.

    print('Fill data ...')
    feed_dict = fill_feed_dict(test_batches,
                               test_labels,
                               params.train_both,
                               batch_placeholders,
                               label_placeholders,
                               corrupt_placeholder)
    # predictions,labels = sess.run(eval_correct, feed_dict)

    print('Execute graph ...')
    predictions, labels = sess.run(eval_correct, feed_dict)

    print('Calculate precision ...')
    for i in range(len(predictions[0])):
        if predictions[0][i] > 0 and labels[0][i] == 1:
            true_count += 1.0
        elif predictions[0][i] < 0 and labels[0][i] == -1:
            true_count += 1.0

    precision = float(true_count) / float(num_examples)

    return precision


def get_thresholds():
    dev_data = ntn_input.load_dev_data()
    entities_list = ntn_input.load_entities(params.data_path)
    relations_list = ntn_input.load_entities(params.data_path)

    num_entities = len(entities_list)
    num_relations = len(relations_list)

    slice_size = params.slice_size
    (init_word_embeds, entity_to_wordvec) = ntn_input.load_init_embeds(params.data_path)

    batch_placeholder = tf.placeholder(tf.float32, shape=(4, batch_size))
    corrupt_placeholder = tf.placeholder(tf.bool, shape=(1))  # Which of e1 or e2 to corrupt?
    predictions_list = ntn.g_function(batch_placeholder, corrupt_placeholder, init_word_embeds, entity_to_wordvec,
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


def getPredictions():
    best_thresholds = get_thresholds()
    test_data = ntn_input.load_test_data()
    entities_list = ntn_input.load_entities(params.data_path)
    relations_list = ntn_input.load_entities(params.data_path)

    num_entities = len(entities_list)
    num_relations = len(relations_list)

    slice_size = params.slice_size
    (init_word_embeds, entity_to_wordvec) = ntn_input.load_init_embeds(params.data_path)

    batch_placeholder = tf.placeholder(tf.float32, shape=(4, batch_size))
    corrupt_placeholder = tf.placeholder(tf.bool, shape=(1))  # Which of e1 or e2 to corrupt?
    predictions_list = ntn.g_function(batch_placeholder, corrupt_placeholder, init_word_embeds, entity_to_wordvec,
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
