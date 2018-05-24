import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd
import random

# import utils

""" Simple linear regression example in TensorFlow
This program tries to predict the number of thefts from
the number of fire in the city of Chicago
Author: Chip Huyen
Prepared for the class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu
"""

DATA_FILE = '../data/fire_theft/fire_theft.xls'
MODEL_PATH = '../data/checkpoints/best_validation'

# Step 1: read in data from the .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

print('type of data: {}, shape of data: {}, number of sample: {}'.format(str(type(data)),
                                                                         str(data.shape),
                                                                         n_samples))

# Step 2: create placeholders for input placeholder_X (number of fire) and label placeholder_Y (number of theft)
placeholder_X = tf.placeholder(tf.float32, name='placeholder_X')
placeholder_Y = tf.placeholder(tf.float32, name='placeholder_Y')

# Step 3: create weight and bias, initialized to 0
w = tf.Variable(0.0, name='weights')
b = tf.Variable(0.0, name='bias')

# Step 4: build model to predict placeholder_Y
Y_predicted = placeholder_X * w + b

# Step 5: use the square error as the loss function
loss = tf.square(placeholder_Y - Y_predicted, name='loss')
# loss = utils.huber_loss(placeholder_Y, Y_predicted)

# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    sess.run(tf.global_variables_initializer())
    
    writer = tf.summary.FileWriter('./graphs/linear_reg', sess.graph)
    saver = tf.train.Saver(tf.trainable_variables())
    
    # Step 8: train the model
    for i in range(100):  # train the model 100 epochs
        total_loss = 0
        for x, y in data:
            # Session runs train_op and fetch values of loss
            _, loss_value = sess.run([optimizer, loss], feed_dict={placeholder_X: x, placeholder_Y: y})
            total_loss += loss_value
        
        # save model
        print('Epoch {0}: {1}'.format(i, total_loss / n_samples))
    
    saver.save(sess, MODEL_PATH + '.sess')
    # close the writer when you're done using it
    writer.close()
    
    # Step 9: output the values of w and b
    w, b = sess.run([w, b])
    print('optimizing w and b: {} - {}'.format(w, b))

# plot the results
placeholder_X, placeholder_Y = data.T[0], data.T[1]
plt.plot(placeholder_X, placeholder_Y, 'bo', label='Real data')
plt.plot(placeholder_X, placeholder_X * w + b, 'r', label='Predicted data')
plt.legend()
plt.show()
