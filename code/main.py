import params
import scipy.io as sio
import numpy as np
import csv
import random
import tensorflow as tf
from tensorflow.python.client import device_lib


# TODO: why do we need initEmbed?
# TODO: what is the content of initEmbed?


def write_words(words):
    file_path = '../data/Wordnet/words.csv'
    
    rand_num = random.randint(0, words.shape[1])
    
    print('Writing file ...')
    with open(file_path, 'w', newline='') as f:
        for i in range(words.shape[1]):
            writer = csv.writer(f)
            
            if rand_num == i:
                print(words[0][i])
            
            writer.writerows([words[0][i]])
    
    print('Finish writing words ...')


def logging_device_placement():
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='b')
    c = tf.matmul(a, b)
    
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    
    print(sess.run(c))


def get_list_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


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
    # write_words(words)
    print('the type of words is ' + str(type(words)))
    print(words.shape)
    print('3')
    we = mat_contents['We']  # WTHITS!
    print('the type of we is ' + str(type(we)))
    print(we.shape)
    # TODO: sample of we
    print('sample of we is ' + str(we[6][16]))
    # =========================================================================
    print('4')
    tree = mat_contents['tree']  # WTHITS!
    print('the type fo tree is ' + str(type(tree)))  # numpy array
    print(tree.shape)  # (38696, 1)
    print('sample of tree is ' + str(tree[30000][0][0][0][0][0]))  # [11095 63941]
    print('sample of tree is ' + str(tree[30000][0]))
    print('the type of tree[30000][0][0][0][0][0] is ' + str(type(tree[30000][0][0][0][0][0])))
    print(np.shape(tree[30000][0][0][0][0][0]))
    print('5')
    
    # write_words(words)
    # write_we(we)
    # write_tree(tree)
    
    word_vecs = [[we[j][i] for j in range(params.embedding_size)] for i in range(len(words[0]))]
    print('6')
    print('the type of word_vecs is ' + str(type(word_vecs)) + ' with length of ' + str(len(word_vecs)))
    print('sample of word_vecs: ' + str(word_vecs[66]))
    print('the type of word_vecs[66] is ' + str(type(word_vecs[66])) + ' with length of ' + str(len(word_vecs[66])))
    # TODO: why do we need map?
    print('the length of tree is ' + str(len(tree)))
    # =========================================================================
    # the length of tree is 38696
    entity_words = [map(int, tree[i][0][0][0][0][0]) for i in range(len(tree))]
    print('7')
    print('the type of entity_words is ' + str(type(entity_words)) + ' with length of ' + str(len(entity_words)))
    print('sample of entity_vecs (tra): ' + str(list(entity_words[30000])))
    return word_vecs, entity_words


if __name__ == '__main__':
    print('Processing ...')
    # word_vecs, entity_words = main()
    # logging_device_placement()
    get_list_devices()
    
    print('Done')
