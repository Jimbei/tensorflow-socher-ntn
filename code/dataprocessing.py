import random
import numpy as np


def index_data(data, entity_list, entity_indices, relation_list):
    print('Sample data')
    # filter entity
    fil_entity_list = random.sample(entity_list, 1000)
    # filter triple
    data = filter_data(data, fil_entity_list)
    # filter relation
    print('original relation list: {}'.format(relation_list))
    relation_list = filter_relation(data)
    print('filtering relation list: {}'.format(relation_list))
    # filter entity indices
    fil_entity_indices = filter_entity_indices(entity_list, entity_indices, fil_entity_list)
    # =========================================================================

    indexing_entities = {fil_entity_list[i]: i for i in range(len(fil_entity_list))}
    indexing_relations = {relation_list[i]: i for i in range(len(relation_list))}
    indexing_data = [[indexing_entities[data[i][0]],
                      indexing_relations[data[i][1]],
                      indexing_entities[data[i][2]]]
                     for i in range(len(data))]

    return indexing_data, len(fil_entity_list), len(relation_list), fil_entity_list


def generate_corrupting_batch(batch_size, data, num_entities, corrupt_size, entity_list):
    # TODO continue doing here
    # random_indices = random.sample(range(len(data)), batch_size)
    # corrupting_batch = [(data[i][0],  # data[i][0] = e1
    #                      data[i][1],  # data[i][1] = r
    #                      data[i][2],  # data[i][2] = e2
    #                      random.randint(0, num_entities - 1))  # random = e3 (corrupted)
    #                     for i in random_indices for _ in range(corrupt_size)]

    return corrupting_batch


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


def filter_data(data, filtering_entity):
    filtering_data = []

    for i in range(len(data)):
        e1, r, e2 = data[i]
        if e1 in filtering_entity and e2 in filtering_entity:
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

    assert len(fil_entity_list) == len(fil_entity_indices)
    return fil_entity_indices


def filter_relation(data):
    filtering_relation = []

    for i in data:
        e1, r, e2 = i
        if r not in filtering_relation:
            filtering_relation.append(r)

    return filtering_relation
