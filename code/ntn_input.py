# Load entity, relation data, precomputed entity vectors based on specified database
# Currently supports FreeBase and Wordnet data
# Author: Dustin Doss

import params
import scipy.io as sio
import numpy as np

import csv

entities_string = '/entities.txt'
relations_string = '/relations.txt'
embeds_string = '/initEmbed.mat'
training_string = '/train.txt'
test_string = '/test.txt'
dev_string = '/dev.txt'


# input: path of dataset to be used
# output: python list of entities in dataset
def load_entities(data_path=params.data_path):
    entities_file = open(data_path + entities_string)
    entities_list = entities_file.read().strip().split('\n')
    entities_file.close()
    return entities_list


# input: path of dataset to be used
# output: python list of relations in dataset
def load_relations(data_path=params.data_path):
    relations_file = open(data_path + relations_string)
    relations_list = relations_file.read().strip().split('\n')
    relations_file.close()
    return relations_list


# input: path of dataset to be used
# output: python dict from entity string->1x100 vector embedding of entity as precalculated
def load_init_embeds(data_path=params.data_path):
    embeds_path = data_path + embeds_string
    return load_embeds(embeds_path)


# input: Generic function to load embeddings from a .mat file
def load_embeds(file_path):
    """
    Load initEmbed
    :param file_path: directory stores initEmbed object
    :return:
        word_vecs:      a list of 67447 initial vectors with the size of 100 elements for each
        entity_word:    a list of 38696 triples encoded in number
    """
    mat_contents = sio.loadmat(file_path)
    words = mat_contents['words']
    we = mat_contents['We']
    tree = mat_contents['tree']

    word_vecs = [[we[j][i] for j in range(params.embedding_size)] for i in range(len(words[0]))]
    entity_words = [map(int, tree[i][0][0][0][0][0]) for i in range(len(tree))]
    # # Debug section
    # print('type of mat_contents is ' + str(type(mat_contents)))
    # # type of mat_contents is dict
    # TODO: write files
    # print('Writing tree.csv file ...')
    # with open('../data/Wordnet/additionalFiles/tree.csv', 'w', newline='') as f:
    #     print('type of tree is ' + str(type(tree)))
    #     # type of tree is <class 'numpy.ndarray'>
    #     print('dimension of tree is ' + str(tree.shape))
    #     for i in range(len(tree)):
    #         writer = csv.writer(f)
    #         writer.writerow(tree[i][0][0][0][0][0])
    # print('Finish writing tree.csv')
    #
    # print('Writing entity_words.csv ...')
    # with open('../data/Wordnet/additionalFiles/entity_words.csv', 'w', newline='') as f:
    #     print('type of entity_words is ' + str(type(entity_words)))
    #     # type of entity_words is <class 'list'>
    #     np_entity_words = np.array(entity_words)
    #     print('dimension of entity_words is ' + str(np_entity_words.shape))
    #     for i in range(len(entity_words)):
    #         writer = csv.writer(f)
    #         writer.writerow(entity_words[i])
    # print('Finish writing entity_words.csv')
    #
    # print('type of word_vecs is ' + str(type(word_vecs)))
    # np_word_vecs = np.array(word_vecs)
    # print('dimension of word_vecs is ' + str(np_word_vecs.shape))
    # print('len(word_vecs) is ' + str(len(word_vecs)))
    # # type of word_vecs is <class 'list'>
    # for i in range(len(word_vecs)):
    #     if i == 100:
    #         print(word_vecs[i])
    #         break
    #
    # for i in range(len(entity_words)):
    #     if i == 99:
    #         print(entity_words[i])
    #         # <map object at 0x00000224227630B8>
    #         print(list(entity_words[i]))
    #         # [50029, 50006, 50004]
    #         break
    #
    # exit()
    # =========================================================================
    return word_vecs, entity_words


def load_training_data(data_path=params.data_path):
    training_file = open(data_path + training_string)
    training_data = [line.split('\t') for line in training_file.read().strip().split('\n')]
    return np.array(training_data)


def load_dev_data(data_path=params.data_path):
    dev_file = open(data_path + test_string)
    dev_data = [line.split('\t') for line in dev_file.read().strip().split('\n')]
    return np.array(dev_data)


def load_test_data(data_path=params.data_path):
    test_file = open(data_path + test_string)
    test_data = [line.split('\t') for line in test_file.read().strip().split('\n')]
    return np.array(test_data)
