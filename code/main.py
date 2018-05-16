def main():
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    entity_indices = [[1], [1, 3], [3, 7], [9, 3, 7], [7]]

    filtering_data = []
    filtering_entity = []
    for i in entity_indices:
        for j in i:
            if j not in filtering_entity:
                filtering_entity.append(j)

    for i in range(len(data)):
        e1, r, e2 = data[i]
        print('e1: {} - e2: {}'.format(e1, e2))
        if e1 in filtering_entity and e2 in filtering_entity:
            filtering_data.append(data[i])

    print(filtering_data)


if __name__ == '__main__':
    main()
