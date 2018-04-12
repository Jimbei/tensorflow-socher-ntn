import params
import scipy.io as sio


# TODO: why do we need initEmbed?
# TODO: what is the content of initEmbed?

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

    word_vecs = [[we[j][i] for j in range(params.embedding_size)] for i in range(len(words[0]))]
    print('6')
    entity_words = [map(int, tree[i][0][0][0][0][0]) for i in range(len(tree))]
    print('7')

    return word_vecs, entity_words


if __name__ == '__main__':
    print('Processing ...')
    word_vecs, entity_words = main()
    
    [[A[i][j]]]
    print('Done')
