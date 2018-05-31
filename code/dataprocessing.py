import random
import numpy as np
import ntn_input
import params
import csv


def index_data(data, entity_list, entity_indices):
    print('Sample data')
    # filter entity
    fil_entity_list = random.sample(entity_list, 2000)
    # filter triple
    data = filter_data(data, fil_entity_list)
    # filter relation
    # print('original relation list: {}'.format(relation_list))
    fil_relation_list = filter_relation(data)
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


def generate_corrupting_batch(batch_size, data, num_entities, corrupt_size, num_relations):
    # sort data according to relation
    # print('\n===== Data =====\n{}\n'.format(data))
    sorted_data = [[T for T in data if r == T[1]] for r in range(num_relations)]
    # print('===== Sort Data =====\n')
    # for T in sorted_data:
    #     print('{} - number of T is {}'.format(T, len(T)))
    # random sample from relation
    batch_size = int(batch_size / num_relations)
    # print('batch_size for sorted data: {}'.format(batch_size))
    random_data = []
    for T in sorted_data:
        if len(T) > batch_size:
            T = random.sample(T, batch_size)

        random_data.append(T)

    # print('\nRandom batch')
    # for T in random_data:
    #     print(T)

    # add corrupting entity
    corrupting_data = []
    for r in random_data:
        for T in r:
            for i in range(corrupt_size):
                corrupting_data.append([T[0], T[1], T[2], random.randint(0, num_entities - 1)])

    # print('\nCorrupting data')
    # for T in corrupting_data:
    #     print(T)

    # random_indices = random.sample(range(len(data)), batch_size)
    # corrupting_batch = [(data[i][0],  # data[i][0] = e1
    #                      data[i][1],  # data[i][1] = r
    #                      data[i][2],  # data[i][2] = e2
    #                      random.randint(0, num_entities - 1))  # random = e3 (corrupted)
    #                     for i in random_indices for _ in range(corrupt_size)]

    return corrupting_data


def split_corrupting_batch(data_batch, num_relations):
    corrupting_entities = [[] for i in range(num_relations)]
    for e1, r, e2, e3 in data_batch:
        # corrupting_entities[r].append((e1, e2, e3))
        corrupting_entities[r].append([e1, e2, e3])

    return corrupting_entities


def fill_feed_dict(relation_batches, train_both, placeholder_data, placeholder_label, placeholder_corrupt):
    feed_dict = {placeholder_corrupt: [train_both and np.random.random() > 0.5]}

    for i in range(len(placeholder_data)):
        feed_dict[placeholder_data[i]] = relation_batches[i]
        feed_dict[placeholder_label[i]] = [[0.0] for j in range(len(relation_batches[i]))]

    # print(feed_dict)
    return feed_dict


def filter_data(data, fil_entity_list):
    filtering_data = []

    for i in range(len(data)):
        e1, r, e2 = data[i]
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


def filter_relation(data):
    filtering_relation = []

    for i in data:
        e1, r, e2 = i
        if r not in filtering_relation:
            filtering_relation.append(r)

    return filtering_relation


def generate_data(mode):
    fil_T = []

    if mode == 0:
        print('In training mode')
    if mode == 1:
        print('In testing mode')
        T = list(ntn_input.load_test_data(params.data_path))
        print('shape of T: {}'.format(np.array(T).shape))
        R = ntn_input.load_relations(params.data_path)
        T = [[t for t in T if r == t[1]] for r in R]
        for t in T:
            t = random.sample(t, 20)
            fil_T.append(t)

        with open(params.data_path + '/filtertest.txt', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            for t in fil_T:
                for i in t:
                    writer.writerow(i)
    return fil_T
