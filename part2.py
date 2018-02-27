#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import os
import util
from model import *

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/work/cse496dl/shared/homework/02/SAVEE-British/', 'directory where FMNIST is located')
flags.DEFINE_string('save_dir', '/work/cse496dl/asturtz', 'directory where model graph and weights are saved')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('max_epoch_num', 200, '')
FLAGS = flags.FLAGS

FILES = [ ('train_x_1.npy', 'train_y_1.npy', 'test_x_1.npy', 'test_y_1.npy')
        , ('train_x_2.npy', 'train_y_2.npy', 'test_x_2.npy', 'test_y_2.npy')
        , ('train_x_3.npy', 'train_y_3.npy', 'test_x_3.npy', 'test_y_3.npy')
        , ('train_x_4.npy', 'train_y_4.npy', 'test_x_4.npy', 'test_y_4.npy')
        ]

MODELS = { 'model-1' : model_conv_2 }

# train_images_name = ['train_x_1.npy','train_x_2.npy','train_x_3.npy','train_x_4.npy']
# train_labels_label = ['train_y_1.npy','train_y_2.npy','train_y_3.npy','train_y_4.npy']

# test_images_name = ['test_x_1.npy','test_x_2.npy','test_x_3.npy','test_x_4.npy']
# test_images_label = ['test_y_1.npy','test_y_2.npy','test_y_3.npy','test_y_4.npy']


def load_data(files):
    load = lambda f: np.load(FLAGS.data_dir + f)
    train_data = reshape(load(files[0]))
    train_labels = load(files[1])
    test_data = reshape(load(files[2]))
    test_labels = load(files[3])
    return (train_data, train_labels), (test_data, test_labels)

def save(file_name, model_name, data):
    np.save(os.path.join(FLAGS.save_dir, model_name, file_name), data)

def reshape(arr):
    return np.reshape(arr, [-1,129,129,1])

def init_graph(model, reg):
    input_placeholder = tf.reshape(
        tf.placeholder(tf.float32, [None, 16641], name='input_placeholder'),
        [-1,129,129,1]
    )
    output = tf.identity(model(input_placeholder), name='output')
    y = tf.placeholder(tf.float32, [None, 7], name='label')

    return input_placeholder, output, y

def minimize_loss(inputs, outputs, labels, reg, total_loss):
    global_step = tf.get_variable(
        'global_step',
        trainable=False,
        shape=[],
        initializer=tf.zeros_initializer
    )
    optimizer = tf.train.AdamOptimizer()
    return global_step, optimizer.minimize(total_loss, global_step=global_step)

def loss(inputs, outputs, labels, reg):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=labels)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    return tf.reduce_mean(cross_entropy + reg * sum(reg_losses))

def batch(data, index):
    return data[index*FLAGS.batch_size:(index+1)*FLAGS.batch_size, :]

def train(data, inputs, y, train_op, total_loss, session):
    ce_vals = []
    train_data, labels = data
    for i in range(train_data.shape[0] // FLAGS.batch_size):
        batch_xs = batch(data[0], i)
        batch_ys = batch(data[1], i)
        _, train_ce = session.run([train_op, total_loss], {inputs: batch_xs, y: batch_ys})
        ce_vals.append(train_ce)

    avg_train_ce = sum(ce_vals) / len(ce_vals)
    return avg_train_ce, ce_vals

def test(data, inputs, y, confusion_matrix_op, total_loss, session):
    ce_vals_v = []
    conf_mxs_v = []
    test_data, labels = data
    for i in range(test_data.shape[0] // FLAGS.batch_size):
        batch_xsv = batch(data[0], i)
        batch_ysv = batch(data[1], i)
        test_cev, conf_matrix_v, _ = session.run([total_loss, confusion_matrix_op], {inputs: batch_xsv, y: batch_ysv})
        ce_vals_v.append(test_cev)
        conf_mxs_v.append(conf_matrix_v)
    avg_test_cev = sum(ce_vals_v) / len(ce_vals_v)
    confusion_sum = sum(conf_mxs_v)
    return avg_test_cev, confusion_sum


def main(argv):
    model_name = argv[1]
    reg_coefficient = float(argv[2])

    inputs, outputs, y = init_graph(MODELS[model_name], reg_coefficient)
    total_loss = loss(inputs, outputs, y, reg_coefficient)
    global_step, train_op = minimize_loss(inputs, outputs, y, reg_coefficient, total_loss)
    confusion_matrix_op = tf.confusion_matrix(tf.argmax(y, axis=1), tf.argmax(outputs, axis=1), num_classes=7)
    saver = tf.train.Saver()

    for files in FILES:
        print('This is start')
        print(files[0])
        train_data, test_data = load_data(files)
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            ce_vals = ([], [])
            best_test_ce = 0
            for epoch in range(1):
                avg_train_ce, _ = train(train_data, inputs, y, train_op, total_loss, session)
                avg_test_cev, confusion_sum = test(test_data, confusion_matrix_op, total_loss, session)
                ce_vals[0].append(avg_train_ce)
                ce_vals[1].append(avg_test_cev)

                print('TRAIN CROSS ENTROPY: ' + str(avg_train_ce))
                print('VALIDATION CROSS ENTROPY: ' + str(avg_test_cev))
                print('VALIDATION CONFUSION MATRIX:')
                print(str(confusion_sum))

            if avg_test_cev < best_test_ce:
                best_test_ce = avg_test_cev
                save('conf-matrix', model_name, confusion_sum)
                save('train', model_name, ce_vals[0])
                save('validation', model_name, ce_vals[1])
                saver.save(
                    session,
                    os.path.join(FLAGS.save_dir, argv[1], 'savee_homework_2'),
                    global_step=global_step
                )

if __name__ == "__main__":
    tf.app.run()
