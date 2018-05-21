import random
import tensorflow as tf
import numpy as np

CKPT_DIR = '../data/checkpoints/best_validation'


def main():

    placeholder_X = tf.placeholder(tf.float32, name='placeholder_X')
    placeholder_Y = tf.placeholder(tf.float32, name='placeholder_Y')

    w = tf.Variable(0.0)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(CKPT_DIR + '.sess.meta')
        saver.restore(sess, CKPT_DIR + '.sess')


if __name__ == '__main__':
    main()
