import ntn_input
import ntn
import params
import tensorflow as tf
import random


def index_data(data, entities, relations):
    entity_to_index = {entities[i]: i for i in range(len(entities))}
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
    test_data = ntn_input.load_test_data(params.data_path)

    print('Load entity list from entities.txt ...')
    entities_list = ntn_input.load_entities(params.data_path)

    print('Load relation list from relations.txt ...')
    relations_list = ntn_input.load_relations(params.data_path)

    print('Index raw data ...')
    test_data = index_data(test_data, entities_list, relations_list)

    batch_size = len(test_data)
    num_entities = len(entities_list)
    num_relations = len(relations_list)

    slice_size = params.slice_size
    init_word_embeds, entity_to_wordvec = ntn_input.load_init_embeds(params.data_path)
    batches, labels = data_to_relation_sets(test_data, num_relations)

    with tf.Graph().as_default():
        sess = tf.Session()

        print('Create placeholders')
        batch_placeholders = [tf.placeholder(tf.float32, shape=(None, 3)) for i in range(num_relations)]
        label_placeholders = [tf.placeholder(tf.float32, shape=(None, 1)) for i in range(num_relations)]

        corrupt_placeholder = tf.placeholder(tf.bool, shape=(1))

        print('Define training model')
        inference = ntn.inference(batch_placeholders,
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

        saved_model = 'Wordnet490'
        path_model = params.output_path + saved_model + '.sess'

        assert path_model == '../output/Wordnet/Wordnet490.sess'

        print('Load training model {}'.format(saved_model))
        saver = tf.train.Saver()
        # saver.restore(sess, params.output_path + 'around100/' + saved_model + '.sess')
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

    # print(debug_printout)
    # print(score_values.get_shape())

    for i in range(len(score_values[0])):
        if score_values[0][i] > 0 and labels[0][i] == 1:
            true_count += 1.0
        elif score_values[0][i] < 0 and labels[0][i] == -1:
            true_count += 1.0
    precision = float(true_count) / float(n_samples)

    print('precision: {}'.format(precision))
    sess.close()
    return precision


def getThresholds():
    dev_data = ntn_input.load_dev_data()
    entities_list = ntn_input.load_entities(params.data_path)
    relations_list = ntn_input.load_entities(params.data_path)

    num_entities = len(entities_list)
    num_relations = len(relations_list)

    slice_size = params.slice_size
    (init_word_embeds, entity_to_wordvec) = ntn_input.load_init_embeds(params.data_path)

    batch_placeholder = tf.placeholder(tf.float32, shape=(4, batch_size))
    corrupt_placeholder = tf.placeholder(tf.bool, shape=(1))  # Which of e1 or e2 to corrupt?
    predictions_list = ntn.inference(batch_placeholder, corrupt_placeholder, init_word_embeds, entity_to_wordvec,
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
    best_thresholds = getThresholds()
    test_data = ntn_input.load_test_data()
    entities_list = ntn_input.load_entities(params.data_path)
    relations_list = ntn_input.load_entities(params.data_path)

    num_entities = len(entities_list)
    num_relations = len(relations_list)

    slice_size = params.slice_size
    (init_word_embeds, entity_to_wordvec) = ntn_input.load_init_embeds(params.data_path)

    batch_placeholder = tf.placeholder(tf.float32, shape=(4, batch_size))
    corrupt_placeholder = tf.placeholder(tf.bool, shape=(1))  # Which of e1 or e2 to corrupt?
    predictions_list = ntn.inference(batch_placeholder, corrupt_placeholder, init_word_embeds, entity_to_wordvec,
                                     num_entities, num_relations, slice_size, batch_size)

    predictions = tf.zeros((test_data.shape[0], 1))
    for i in range(test_data.shape[0]):
        # get relation
        rel = test_data[i, 1]

        # get labels based on predictions
        if predictions_list[i, 0] <= best_thresholds[rel, 0]:
            predictions[i, 0] = 1
        else:
            predictions[i, 0] = -1

    return predictions


if __name__ == "__main__":
    run_evaluation()
