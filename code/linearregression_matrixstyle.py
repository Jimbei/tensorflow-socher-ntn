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
    return features * w + b


def fill_feed_dict(data, features_plac, labels_plac):
    X = np.reshape(data[:, 0], (len(data), 1))
    Y = np.reshape(data[:, 1], (len(data), 1))

    return {features_plac: X, labels_plac: Y}


def main():
    print('Load data')
    train_data, n_samples = load_data()

    print('Define placeholders')
    features = tf.placeholder(tf.float32, shape=(None, 1), name='sample_')
    labels = tf.placeholder(tf.float32, shape=(None, 1), name='label_')

    print('Define variables')
    w = tf.Variable(0.0, name='weights')
    b = tf.Variable(0.0, name='bias')

    print('Define hypothesis function')
    pred_labels = hypothesis(features, w, b)

    print('Define loss function')
    loss = tf.reduce_mean(tf.square(labels - pred_labels, name='loss'))

    print('Define optimizer function')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        X = np.reshape(train_data[:, 0], (len(train_data), 1))
        Y = np.reshape(train_data[:, 1], (len(train_data), 1))
        feed_dict = {features: X, labels: Y}

        for i in range(100):
            _, loss_value = sess.run([optimizer, loss], feed_dict)
            print('Epoch {} has loss value {}'.format(i, loss_value / len(train_data)))

        w, b = sess.run([w, b])
        print('optimizing w and b: {} - {}'.format(w, b))

    features, labels = train_data.T[0], train_data.T[1]
    plt.plot(features, labels, 'bo', label='Real data')
    plt.plot(features, features * w + b, 'r', label='Predicted data')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
