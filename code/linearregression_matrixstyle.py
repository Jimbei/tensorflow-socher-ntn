import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import xlrd

DATA_DIR = '../data/fire_theft/fire_theft.xls'
CKPT_DIR = '../data/checkpoints/best_validation'


def load_data():
    book = xlrd.open_workbook(DATA_DIR, encoding_override="utf-8")
    sheet = book.sheet_by_index(0)
    data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
    n_samples = len(data)
    
    return data, n_samples


def hypothesis(features, w, b):
    tmp = w * features
    tmp = tmp + b
    return tmp


def fill_feed_dict(data, features, labels):
    feed_dict = {}
    
    for i in range(len(features)):
        feed_dict[features[i]] = data[i, 0]
        feed_dict[labels[i]] = data[i, 1]
    
    return feed_dict


def main():
    print('Load data')
    train_data, n_samples = load_data()
    
    print('Define placeholders')
    features = [tf.placeholder(tf.float32, shape=(), name='sample_' + str(i))
                for i in range(n_samples)]
    labels = [tf.placeholder(tf.float32, shape=(), name='label_' + str(i))
              for i in range(n_samples)]
    
    print('Define variables')
    w = tf.Variable(tf.zeros(0.0, tf.float32))
    b = tf.Variable(tf.zeros(0.0, tf.float32))
    
    print('Define hypothesis function')
    pred_labels = hypothesis(features, w, b)
    
    print('Define loss function')
    loss = tf.square(labels - pred_label, name='loss')
    
    print('Define optimizer function')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.trainable_variables())
        feed_dict = fill_feed_dict(train_data, features, labels)
        
        for i in range(100):
            __, loss_value = sess.run([optimizer, loss], feed_dict)
            print('Epoch {} has loss value {}'.format(i, loss_value))
            if i == 99:
                saver.save(sess, CKPT_DIR)


if __name__ == '__main__':
    main()
