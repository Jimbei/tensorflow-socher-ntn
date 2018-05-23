import numpy as np
import tensorflow as tf


def main():
    a = tf.constant([1, 2, 3, 4], name='a')
    b = tf.constant(2, name='b')
    c = a * b
    
    with tf.Session() as sess:
        print(sess.run(c))


if __name__ == '__main__':
    main()
