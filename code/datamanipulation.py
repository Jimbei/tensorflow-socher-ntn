import random
import scipy.io as sio
import numpy as np
import params
import csv
import random


def index_data(data, entity_list, entity_indices):
    print('Sample data')
    # filter entity
    fil_entity_list = random.sample(entity_list, 2000)
    # filter triple
    data = filter_data(data, fil_entity_list, 0)
    # filter relation
    # print('original relation list: {}'.format(relation_list))
    fil_relation_list = filter_relation(data, 0)
    # print('filtering relation list: {}'.format(fil_relation_list))
    # filter entity indices
    fil_entity_indices = filter_entity_indices(entity_list, entity_indices, fil_entity_list)
    # =========================================================================

    indexing_entities = {fil_entity_list[i]: i for i in range(len(fil_entity_list))}
    indexing_relations = {fil_relation_list[i]: i for i in range(len(fil_relation_list))}
    indexing_data = [[indexing_entities[data[i][0]],
                      indexing_relations[data[i][1]],
                      indexing_entities[data[i][2]]]
                     for i in range(len(data))]

    num_entities = len(fil_entity_list)
    num_relations = len(fil_relation_list)

    return indexing_data, fil_entity_indices, num_entities, num_relations


def generate_corrupting_batch(batch_size, triples, num_entities, corrupt_size, num_relations):
    sorted_triples = [[T for T in triples if r == T[1]] for r in range(num_relations)]

    batch_size = int(batch_size / num_relations)
    random_data = []
    for t in sorted_triples:
        if len(t) > batch_size:
            t = random.sample(t, batch_size)

        random_data.append(t)

    # add corrupting entity
    corrupting_data = []
    for r in random_data:
        for t in r:
            for i in range(corrupt_size):
                corrupting_data.append([t[0], t[1], t[2], random.randint(0, num_entities - 1)])

    return corrupting_data


def split_corrupting_batch(data_batch, num_relations):
    corrupting_entities = [[] for i in range(num_relations)]
    for e1, r, e2, e3 in data_batch:
        corrupting_entities[r].append([e1, e2, e3])

    return corrupting_entities


def fill_feed_dict(relation_batches, train_both, placeholder_data, placeholder_label, placeholder_corrupt):
    feed_dict = {placeholder_corrupt: [train_both and np.random.random() > 0.5]}

    for i in range(len(placeholder_data)):
        feed_dict[placeholder_data[i]] = relation_batches[i]
        feed_dict[placeholder_label[i]] = [[0.0] for j in range(len(relation_batches[i]))]

    # print(feed_dict)
    return feed_dict


def filter_data(data, fil_entity_list, mode):
    filtering_data = []

    for i in range(len(data)):
        if mode == 0:
            e1, r, e2 = data[i]
        if mode == 1:
            e1, r, e2, label = data[i]
        if e1 in fil_entity_list and e2 in fil_entity_list:
            filtering_data.append(data[i])

    return filtering_data


def filter_entity(entity_indices):
    filtering_entity = []

    for i in entity_indices:
        for j in i:
            if j not in filtering_entity:
                filtering_entity.append(j)

    return filtering_entity


def filter_entity_indices(entity_list, entity_indices, fil_entity_list):
    fil_entity_indices = []

    for i in range(len(entity_list)):
        if entity_list[i] in fil_entity_list:
            fil_entity_indices.append(entity_indices[i])

    return fil_entity_indices


def filter_relation(triples, mode):
    filtering_relation = []

    for t in triples:
        if mode == 0:
            e1, r, e2 = t
        if mode == 1:
            e1, r, e2, label = t
        if r not in filtering_relation:
            filtering_relation.append(r)

    return filtering_relation


def load_init_embeds(data_path):
    file_path = data_path + '/initEmbed.mat'
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


def load_entities(data_path=params.DATA_DIR):
    entities_file = open(data_path + '/entities.txt')
    entities_list = entities_file.read().strip().split('\n')
    entities_file.close()
    return entities_list


def load_relations(data_path=params.DATA_DIR):
    relations_file = open(data_path + '/relations.txt')
    relations_list = relations_file.read().strip().split('\n')
    relations_file.close()
    return relations_list
