import numpy as np
import tensorflow as tf

import hypothesis
import ntn_input
import params

saved_model = 'Wordnet490'
CKPT_DIR = params.output_path + saved_model + '.sess'


def index_data(data, entities, relations):
    entity_to_index = {entities[i]: i for i in range(len(entities))}
    relation_to_index = {relations[i]: i for i in range(len(relations))}
    indexed_data = [(entity_to_index[data[i][0]],
                     relation_to_index[data[i][1]],
                     entity_to_index[data[i][2]],
                     float(data[i][3]))
                    for i in range(len(data))]

    return indexed_data


def fill_feed_dict(batches, labels, train_both, data_plah, label_plah, corrupt_plah):
    feed_dict = {corrupt_plah: [train_both and np.random.random() > 0.5]}
    for i in range(len(data_plah)):
        feed_dict[data_plah[i]] = batches[i]
    for i in range(len(label_plah)):
        feed_dict[label_plah[i]] = labels[i]
    return feed_dict


# dataset is in the form (e1, R, e2, label)
def data_to_relation_sets(data_batch, num_relations):
    batches = [[] for _ in range(num_relations)]
    labels = [[] for _ in range(num_relations)]

    for e1, r, e2, label in data_batch:
        batches[r].append([e1, e2, 1])
        labels[r].append([label])

    return batches, labels


def run_evaluation():
    print('Begin evaluation process')
    print('Load data from test.txt ...')
    triples = ntn_input.load_test_data(params.data_path)

    print('Load entity list from entities.txt ...')
    entities_list = ntn_input.load_entities(params.data_path)

    print('Load relation list from relations.txt ...')
    relations_list = ntn_input.load_relations(params.data_path)

    print('Index raw data ...')
    indexing_triples = index_data(triples, entities_list, relations_list)

    batch_size = len(indexing_triples)
    n_entities = len(entities_list)
    n_relations = len(relations_list)

    slice_size = params.slice_size
    init_word_embeds, entity_to_wordvec = ntn_input.load_init_embeds(params.data_path)
    batches, labels = data_to_relation_sets(indexing_triples, n_relations)

    # sess = tf.Session()

    print('Create placeholders')
    data_plah = [tf.placeholder(tf.float32, shape=(None, 3), name='dataplah_' + str(i)) for i in range(n_relations)]
    label_plah = [tf.placeholder(tf.float32, shape=(None, 1), name='labelplah_' + str(i)) for i in range(n_relations)]
    corrupt_plah = tf.placeholder(tf.bool, shape=(1))

    E = tf.Variable(init_word_embeds)
    W = [tf.Variable(tf.truncated_normal([params.embedding_size, params.embedding_size, slice_size]))
         for _ in range(n_relations)]
    V = [tf.Variable(tf.zeros([slice_size, 2 * params.embedding_size]))
         for _ in range(n_relations)]
    b = [tf.Variable(tf.zeros([slice_size, 1]))
         for _ in range(n_relations)]
    U = [tf.Variable(tf.ones([1, slice_size]))
         for _ in range(n_relations)]

    print('Define hypothesis function')
    values_labels = hypothesis.hypothesis(data_plah,
                                          corrupt_plah,
                                          init_word_embeds,
                                          entity_to_wordvec,
                                          n_entities,
                                          n_relations,
                                          slice_size,
                                          batch_size,
                                          True,
                                          label_plah,
                                          E, W, V, b, U)
    eval_correct = hypothesis.eval(values_labels)
    print('Shape of eval_correct: {}'.format(np.shape(eval_correct)))

    with tf.Session() as sess:
        print('Load checkpoint {}'.format(saved_model))
        saver = tf.train.Saver()
        saver.restore(sess, params.output_path + saved_model + '.sess')

        feed_dict = fill_feed_dict(batches,
                                   labels,
                                   params.train_both,
                                   data_plah,
                                   label_plah,
                                   corrupt_plah)

        precision, min_score, max_score = do_eval(sess,
                                                  eval_correct,
                                                  feed_dict,
                                                  batch_size)

        print('precision: {} - min score: {} - max score: {}'.format(precision, min_score, max_score))


def do_eval(sess,
            eval_correct,
            feed_dict,
            batch_size):
    score_values, labels = sess.run(eval_correct, feed_dict)

    true_count = 0.
    for i in range(len(score_values[0])):
        if score_values[0][i] > 0 and labels[0][i] == 1:
            true_count += 1.0
        elif score_values[0][i] < 0 and labels[0][i] == -1:
            true_count += 1.0

    precision = float(true_count) / float(batch_size)
    min_score = min(score_values[0])
    max_score = max(score_values[0])

    return precision, min_score, max_score
