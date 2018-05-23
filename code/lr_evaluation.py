import random
import tensorflow as tf
import numpy as np
import xlrd

CKPT_DIR = '../data/checkpoints/best_validation'
DATA_DIR = '../data/fire_theft/fire_theft.xls'


def load_data():
    book = xlrd.open_workbook(DATA_DIR, encoding_override="utf-8")
    sheet = book.sheet_by_index(0)
    data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
    n_samples = sheet.nrows - 1
    
    return data, n_samples


def hypothesis(w, b, feature):
    return feature * w + b


def eval_hypothesis(sess, feature, label, pred_label):
    error = 0
    
    sess.run(pred_label, feed_dict={})
    
    # load test data
    test_data, n_samples = load_data()
    
    for x, y in test_data:
    
    for i in range(len(Y)):
        error = error + abs(Y - predicting_Y)
    
    return error


def main():
    # refedine placeholders
    feature = tf.placeholder(tf.float32, name='feature')
    label = tf.placeholder(tf.float32, name='label')
    
    # redefine variables
    w = tf.Variable(0.0, name='weights')
    b = tf.Variable(0.0, name='bias')
    
    # define hypothesis function
    pred_label = hypothesis(w, b, feature)
    
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(CKPT_DIR + '.sess.meta')
        saver.restore(sess, CKPT_DIR + '.sess')
        
        eval_hypothesis(sess, feature, label, pred_label)


if __name__ == '__main__':
    main()
