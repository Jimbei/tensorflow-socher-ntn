import os
import tensorflow as tf
import scipy.io as sio
import numpy as np

data_path = '../data/Wordnet'

entities_string = '/entities.txt'
relations_string = '/relations.txt'
embeds_string = '/initEmbed.mat'
training_string = '/train.txt'
test_string = '/test.txt'
dev_string = '/dev.txt'


def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    
    def f1(): return 0.5 * tf.square(residual)
    
    def f2(): return delta * residual - 0.5 * tf.square(delta)
    
    return tf.cond(residual < delta, f1, f2)


def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass


def load_entities():
    entities_file = open(data_path + entities_string)
    entities_list = entities_file.read().strip().split('\n')
    entities_file.close()
    return entities_list


def load_relations(d):
    relations_file = open(data_path + relations_string)
    relations_list = relations_file.read().strip().split('\n')
    relations_file.close()
    return relations_list


def load_init_embeds():
    file_path = data_path + embeds_string
    embedding_size = 100

    mat_contents = sio.loadmat(file_path)
    words = mat_contents['words']
    we = mat_contents['We']
    tree = mat_contents['tree']
    
    word_vecs = [[we[j][i] for j in range(embedding_size)] for i in range(len(words[0]))]
    # entity_words = [map(int, tree[i][0][0][0][0][0]) for i in range(len(tree))]
    entity_words = [list(tree[i][0][0][0][0][0]) for i in range(len(tree))]
    
    return word_vecs, entity_words


def load_training_data():
    training_file = open(data_path + training_string)
    training_data = [line.split('\t') for line in training_file.read().strip().split('\n')]
    # TODO: change the returning type into tf.data
    return np.array(training_data)


def load_dev_data():
    dev_file = open(data_path + test_string)
    dev_data = [line.split('\t') for line in dev_file.read().strip().split('\n')]
    # TODO: change the returning type into tf.data
    return np.array(dev_data)


def load_test_data():
    test_file = open(data_path + test_string)
    test_data = [line.split('\t') for line in test_file.read().strip().split('\n')]
    # TODO: change the returning type into tf.data
    return np.array(test_data)
