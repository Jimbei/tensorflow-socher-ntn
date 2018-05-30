# Load entity, relation data, precomputed entity vectors based on specified database
# Currently supports FreeBase and Wordnet data
# Author: Dustin Doss

import params
import scipy.io as sio
import numpy as np
import random

entities_string = '/entities.txt'
relations_string = '/relations.txt'
embeds_string = '/initEmbed.mat'
training_string = '/train.txt'
test_string = '/test.txt'
dev_string = '/dev.txt'


# input: path of dataset to be used
# output: python list of entities in dataset


# input: path of dataset to be used
# output: python list of relations in dataset
def load_relations(data_path=params.DATA_DIR):
    relations_file = open(data_path + '/relations.txt')
    relations_list = relations_file.read().strip().split('\n')
    relations_file.close()
    return relations_list


# input: path of dataset to be used
# output: python dict from entity string->1x100 vector embedding of entity as precalculated
def load_init_embeds(data_path=params.DATA_DIR):
    embeds_path = data_path + embeds_string
    return load_embeds(embeds_path)


# input: Generic function to load embeddings from a .mat file
def load_embeds(file_path):
    mat_contents = sio.loadmat(file_path)
    words = mat_contents['words']
    we = mat_contents['We']
    tree = mat_contents['tree']
    word_vecs = [[we[j][i] for j in range(params.embedding_size)] for i in range(len(words[0]))]
    entity_indices = [list(map(int, tree[i][0][0][0][0][0])) for i in range(len(tree))]
    
    return word_vecs, entity_indices


def load_triples(data_path, mode):
    if mode == 0:
        data_file = open(data_path + '/train.txt')
    elif mode == 1:
        data_file = open(data_path + '/test.txt')
    else:
        data_file = open(data_path + '/production.txt')
    data = [line.split('\t') for line in data_file.read().strip().split('\n')]
    return np.array(data)


def load_dev_data(data_path=params.DATA_DIR):
    dev_file = open(data_path + test_string)
    dev_data = [line.split('\t') for line in dev_file.read().strip().split('\n')]
    return np.array(dev_data)


def load_test_data(data_path=params.DATA_DIR):
    data_file = open(data_path + '/test.txt')
    data = [line.split('\t') for line in data_file.read().strip().split('\n')]
    
    return np.array(data)
