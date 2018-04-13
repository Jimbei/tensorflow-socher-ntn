import params
import scipy.io as sio
import numpy as np


# TODO: why do we need initEmbed?
# TODO: what is the content of initEmbed?


def write_words(words):
    file_path = '../data/Wordnet/words.csv'
    
    with open(file_path, 'w') as writer:
        pass
    pass


def write_we(we):
    file_path = '../data/Wordnet/words.csv'
    pass


def write_tree(tree):
    file_path = '../data/Wordnet/words.csv'
    pass


def main():
    file_path = '../data/Wordnet/initEmbed.mat'
    print('1')
    mat_contents = sio.loadmat(file_path)
    print('the type of mat_contents is ' + str(type(mat_contents)))
    # TODO: get all the keys of mat_contents
    for key in mat_contents.keys():
        print(key)
    # =========================================================================
    print('2')
    words = mat_contents['words']  # WTHITS!
    print('the type of words is ' + str(type(words)))
    print(words.shape)
    print(words.item(3))
    print(words.item(4))
    print(words.item(5))
    print(words.item(6))
    print(words.item(7))
    for i in range(67447):
        print(words.item(i))
    print('3')
    we = mat_contents['We']  # WTHITS!
    print(we.shape)
    print('4')
    tree = mat_contents['tree']  # WTHITS!
    print(tree.shape)
    print('5')
    
    # write_words(words)
    # write_we(we)
    # write_tree(tree)
    
    word_vecs = [[we[j][i] for j in range(params.embedding_size)] for i in range(len(words[0]))]
    print('6')
    entity_words = [map(int, tree[i][0][0][0][0][0]) for i in range(len(tree))]
    print('7')
    
    return word_vecs, entity_words


if __name__ == '__main__':
    print('Processing ...')
    # word_vecs, entity_words = main()
    mymat1 = np.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    mymat2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print('type of mymat1 is ' + str(type(mymat1)))
    print('type of mymat2 is ' + str(type(mymat2)))
    print()
    
    print('Done')
