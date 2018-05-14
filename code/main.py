import csv
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import xlrd

import utils


# from tensorflow.python.client import device_lib

DATA_FILE = '../data/Wordnet/additionalFiles/fire_theft.xls'

# TODO: why do we need initEmbed?
# This is a kind of pre-trained model used for initial step
# TODO: what is the content of initEmbed?
# initEmbed is a dictionary stores initial information for training
# TODO: understand tf.split()
# tensor,
# TODO: understand tf.squeeze(a_tensor, list_specified_dimension_size_1)
# Remove dimensions of size 1 from the shape of a tensor
# TODO: understand tf.expand_dims(tf.shape(e1v_pos)[1], 0)
# TODO: understand tf.Print()
# print_out_x = tf.Print(x, [x], summarize=x.shape[0], message='Value of x: ', name='x_value')


def lab():
    n_sample = 100
    train_X = np.linspace(1, 50, n_sample)
    train_Y = 1.5 * train_X + 10.0 + np.random.normal(scale=10, size=1)
    # train_Y = [1.5 * train_X[i] + random.randint(1, 6) for i in range(0, len(train_X))]
    # data_vstack = np.vstack((original_X, original_Y)).reshape(n_sample, 2)
    # print(data_vstack)
    # random.shuffle(data_vstack)
    # print(data_vstack)
    #
    # data_zip = list(zip(original_X, original_Y))
    # print(data_zip)
    # random.shuffle(data_zip)
    # print(data_zip)
    #
    # shuffle_X = [data_zip[i][0] for i in range(0, len(original_X))]
    # shuffle_Y = [data_zip[i][1] for i in range(0, len(original_Y))]
    #
    # plt.plot(original_X, original_Y, 'bo', label='Original data')
    # plt.plot(shuffle_X, shuffle_Y, 'r', label='Shuffle data')
    # plt.legend()
    # plt.show()
    #
    # exit()

    data = list(zip(train_X, train_Y))
    random.shuffle(data)
    
    # print('type of train_X: {}, type of train_Y: {}'.format(str(type(train_X)), str(type(train_Y))))
    # print('shape of train_X: {}, shape of train_Y: {}'.format(str(train_X.shape), str(train_Y.shape)))
    # TODO: convert to np.array
    # data = np.vstack((train_X, train_Y)).reshape(n_sample, 2)
    # print(data)
    # exit()
    # data = np.asanyarray([[train_X[i], train_Y[i]] for i in range(len(train_X))])
    
    # book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
    # sheet = book.sheet_by_index(0)
    # data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
    # n_samples = sheet.nrows - 1

    # print('type of data: {}, shape of data: {}'.format(str(type(data)), str(data.shape)))
    # print('sample of data: ' + str(data[random.randint(0, len(data))]))
    # random.shuffle(data)  # have to shuffle data, otherwise w and b would be na
    # =========================================================================
    
    # Step 2
    placeholder_X = tf.placeholder(dtype=tf.float32, name='X')
    placeholder_Y = tf.placeholder(dtype=tf.float32, name='Y')
    
    # Step 3
    w = tf.Variable(0.0, name='weight')
    b = tf.Variable(0.0, name='bias')
    
    # Step 4
    Y_predicted = placeholder_X * w + b
    
    # Step 5
    loss = tf.square(placeholder_Y - Y_predicted, name="loss")
    
    # Step 6
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
    
    with tf.Session() as sess:
        # Step 7
        sess.run(tf.global_variables_initializer())
        
        # Step 8
        for i in range(30):
            total_loss = 0
            for x, y in data:
                # sess.run(optimizer, feed_dict={placeholder_X: x, placeholder_Y: y})
                # print('x: {} and y: {}'.format(x, y))
                _, l = sess.run([optimizer, loss], feed_dict={placeholder_X: x, placeholder_Y: y})
                total_loss += l
            print('Epoch {0}: {1}'.format(i, total_loss / n_sample))

        # Step 9
        w_value, b_value = sess.run([w, b])
        print('optimizing w and b: {} - {}'.format(w_value, b_value))


if __name__ == '__main__':
    print('Hello World\n')
    # inference()
    lab()
    exit()
