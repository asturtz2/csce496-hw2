import tensorflow as tf
import numpy as np
import itertools
import os
import util
import model

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/work/cse496dl/shared/homework/02/EMODB-German/', 'directory where FMNIST is located')
flags.DEFINE_string('save_dir', '/work/cse496dl/ebrahim31', 'directory where model graph and weights are saved')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('max_epoch_num', 200, '')
FLAGS = flags.FLAGS

FILES = [ ('train_x_1.npy', 'train_y_1.npy', 'test_x_1.npy', 'test_y_1.npy')
        , ('train_x_2.npy', 'train_y_2.npy', 'test_x_2.npy', 'test_y_2.npy')
        , ('train_x_3.npy', 'train_y_3.npy', 'test_x_3.npy', 'test_y_3.npy')
        , ('train_x_4.npy', 'train_y_4.npy', 'test_x_4.npy', 'test_y_4.npy')
        ]

MODELS = { 'model-1' : model.model_conv_2 }

# train_images_name = ['train_x_1.npy','train_x_2.npy','train_x_3.npy','train_x_4.npy']
# train_labels_label = ['train_y_1.npy','train_y_2.npy','train_y_3.npy','train_y_4.npy']

# test_images_name = ['test_x_1.npy','test_x_2.npy','test_x_3.npy','test_x_4.npy']
# test_images_label = ['test_y_1.npy','test_y_2.npy','test_y_3.npy','test_y_4.npy']


def load_data(files):
    load = lambda f: np.load(FLAGS.data_dir + f)
    train_data = load(files[0])
    train_labels = load(files[1])
    test_data = load(files[2])
    test_labels = load(files[3])
    return reshape(train_data), reshape(train_labels), reshape(test_data), reshape(test_labels)

def save(file_name, model_name, data):
    np.save(os.path.join(FLAGS.save_dir, model_name, file_name), data)

def reshape(tensor):
    return np.reshape(tensor, [-1,129,129,1])

def init_graph(model, reg):
    input_placeholder = reshape(
        tf.placeholder(tf.float32, [None, 16641], name='input_placeholder')
    )
    output = tf.identity(model(input_placeholder), name='output')
    y = tf.placeholder(tf.float32, [None, 7], name='label')

    global_step_tensor = tf.get_variable(
        'global_step',
        trainable=False,
        shape=[],
        initializer=tf.zeros_initializer
    )
    total_loss = loss(input_placeholder, output, y, reg)
    optimizer = tf.train.AdamOptimizer()
    return optimizer.minimize(total_loss, global_step=global_step_tensor)

def minimize_loss(inputs, outputs, labels, reg):
    global_step_tensor = tf.get_variable(
        'global_step',
        trainable=False,
        shape=[],
        initializer=tf.zeros_initializer
    )
    total_loss = loss(inputs, outputs, labels, reg)
    optimizer = tf.train.AdamOptimizer()
    return optimizer.minimize(total_loss, global_step=global_step_tensor)

def loss(inputs, outputs, labels, reg):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=labels)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    return tf.reduce_mean(cross_entropy(outputs, labels) + reg * sum(reg_losses))

def batch(data, index):
    return data[index*FLAGS.batch_size:(index+1)*FLAGS.batch_size, :]

def train(data, labels, train_op):
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        # run training
        lossControl = []
        training_avgs = []
        epoch = -1
        eLogic = True
        while (eLogic == True and epoch <= FLAGS.max_epoch_num):
            epoch += 1
            #for epoch in range(FLAGS.max_epoch_num):
            print('Epoch: ' + str(epoch))
            # run gradient steps and report mean loss on train data
            ce_vals = []
            for i in range(train_num_examples // FLAGS.batch_size):
                batch_xs = train_data[i*batch_size:(i+1)*batch_size, :]
                batch_ys = train_labels[i*batch_size:(i+1)*batch_size, :]
                _, train_ce = session.run([train_op, total_loss], {input_placeholder: batch_xs, y: batch_ys})
                ce_vals.append(train_ce)

            avg_train_ce = sum(ce_vals) / len(ce_vals)
            training_avgs.append(avg_train_ce)
            print('TRAIN CROSS ENTROPY: ' + str(avg_train_ce))

            ce_vals_v = []
            conf_mxs_v = []
            for i in range(test_set_num_examples // batch_size):
                batch_xsv = test_images[i*batch_size:(i+1)*batch_size, :]
                batch_ysv = test_labels[i*batch_size:(i+1)*batch_size, :]
                test_cev, conf_matrix_v, _ = session.run([total_loss, confusion_matrix_op, output], {input_placeholder: batch_xsv, y: batch_ysv})
                ce_vals_v.append(test_cev)
                conf_mxs_v.append(conf_matrix_v)
            avg_test_cev = sum(ce_vals_v) / len(ce_vals_v)
            print('VALIDATION CROSS ENTROPY: ' + str(avg_test_cev))
            earlyStoppingParam = 5
            if epoch > earlyStoppingParam :
                if len(lossControl) > earlyStoppingParam :
                    lossControl.pop(0)
                    lossControl.append(avg_test_cev)
                    if (np.average(lossControl)+0.1*np.std(lossControl) < avg_test_cev):
                        print('Early stopping happens at ' + str(epoch))
                        print('the average+1std is: '+str(( np.average(lossControl)+np.std(lossControl))))
                        path_prefix = saver.save(
                        session,
                        os.path.join(FLAGS.save_dir, argv[1], 'homework_1'),
                        global_step=global_step_tensor
                        )
                        saver = tf.train.import_meta_graph(path_prefix + '.meta')
                        eLogic = False
                else :
                    lossControl.append(avg_test_cev)
            print('VALIDATION CONFUSION MATRIX:')
            confusion_sum = sum(conf_mxs_v)
            print(str(confusion_sum))
            print(j)
        Best_set[j] =  avg_test_cev
        print(Best_set)
        print('I am here!')


def main(argv):
    model_name = argv[1]
    reg_coefficient = argv[2]
    inputs, outputs, y = init_graph(MODELS[model_name])
    train_op = minimize_loss(inputs, outputs, y, reg_coefficient)
    confusion_matrix_op = tf.confusion_matrix(tf.argmax(y, axis=1), tf.argmax(output, axis=1), num_classes=7)
    # load data
    Best_set = []
    for files in FILES:
        print('This is start')
        train_data, train_labels, test_data, test_labels = load_data(files)
        test_set_num_examples =  test_images.shape[0]
        train_num_examples = train_data.shape[0]
        # cross_entropy1 = tf.reduce_mean(total_loss)
        saver = tf.train.Saver()
    Index = np.argmin (Best_set)
    train_data = np.load(FLAGS.data_dir + train_images_name[Index])
    train_labels = np.load(FLAGS.data_dir + train_labels_label[Index])
    test_images = np.load(FLAGS.data_dir + test_images_name[Index])
    test_labels = np.load(FLAGS.data_dir + test_images_label[Index])
    test_set_num_examples =  test_images.shape[0]
    train_num_examples = train_data.shape[0]
    train_data = np.reshape(train_data,[-1,129,129,1])
    test_images = np.reshape(test_images,[-1,129,129,1])
    input_placeholder = tf.placeholder(tf.float32, [None, 16641],
            name='input_placeholder')
    input_placeholder = tf.reshape(input_placeholder,[-1,129,129,1])
    #output = models[argv[1]](input_placeholder) model_conv
    output = model.model_conv_2(input_placeholder)
    output = tf.identity(output, name = 'output')
    # define classification loss
    y = tf.placeholder(tf.float32, [None, 7], name='label')

    #with tf.name_scope('cross_entropy') as scope:
    cross_entropy  = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)
    #cross_entropy = tf.reduce_mean(cross_entropy)

        #if argv[1] in ['model_1', 'model_2', 'model_5', 'model_6']:
        #    total_loss = tf.reduce_mean(cross_entropy)
        #else:
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_coeff = 0.0001
    total_loss = tf.reduce_mean(cross_entropy + reg_coeff * sum(regularization_losses))
    # cross_entropy1 = tf.reduce_mean(total_loss)
    confusion_matrix_op = tf.confusion_matrix(tf.argmax(y, axis=1), tf.argmax(output, axis=1), num_classes=7)

    # set up training and saving functionality
    global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(total_loss, global_step=global_step_tensor)
    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        # run training
        batch_size = FLAGS.batch_size
        lossControl = []
        training_avgs = []
        epoch = -1
        eLogic = True
        while (eLogic == True and epoch <= FLAGS.max_epoch_num):
        #for epoch in range(FLAGS.max_epoch_num):
            epoch +=1
            print('Epoch: ' + str(epoch))
            # run gradient steps and report mean loss on train data
            ce_vals = []
            for i in range(train_num_examples // batch_size):
                batch_xs = train_data[i*batch_size:(i+1)*batch_size, :]
                batch_ys = train_labels[i*batch_size:(i+1)*batch_size, :]
                _, train_ce = session.run([train_op, total_loss], {input_placeholder: batch_xs, y: batch_ys})
                ce_vals.append(train_ce)
            avg_train_ce = sum(ce_vals) / len(ce_vals)
            training_avgs.append(avg_train_ce)
            print('TRAIN CROSS ENTROPY: ' + str(avg_train_ce))
            ce_vals_v = []
            conf_mxs_v = []
            for i in range(test_set_num_examples // batch_size):
                batch_xsv = test_images[i*batch_size:(i+1)*batch_size, :]
                batch_ysv = test_labels[i*batch_size:(i+1)*batch_size, :]
                test_cev, conf_matrix_v, _ = session.run([total_loss, confusion_matrix_op, output], {input_placeholder: batch_xsv, y: batch_ysv})
                ce_vals_v.append(test_cev)
                conf_mxs_v.append(conf_matrix_v)
            avg_test_cev = sum(ce_vals_v) / len(ce_vals_v)
            print('VALIDATION CROSS ENTROPY: ' + str(avg_test_cev))

            earlyStoppingParam = 5

            if epoch > earlyStoppingParam :
                if len(lossControl) > earlyStoppingParam :
                    lossControl.pop(0)
                    lossControl.append(avg_test_cev)
                    if (np.average(lossControl)+0.1*np.std(lossControl) < avg_test_cev):
                        print('Early stopping happens at ' + str(epoch))
                        print('the average+1std is: '+str(( np.average(lossControl)+np.std(lossControl))))
                        path_prefix = saver.save(
                            session,
                            os.path.join(FLAGS.save_dir, argv[1], 'homework_1'),
                            global_step=global_step_tensor
                        )
                        saver = tf.train.import_meta_graph(path_prefix + '.meta')
                        eLogic = False
                else :
                    lossControl.append(avg_test_cev)

            print('VALIDATION CONFUSION MATRIX:')
            confusion_sum = sum(conf_mxs_v)
            print(str(confusion_sum))

            np.save(os.path.join(FLAGS.save_dir, argv[1], 'conf-matrix'), confusion_sum)
            np.save(os.path.join(FLAGS.save_dir, argv[1], 'train'),
                    training_avgs)
            np.save(os.path.join(FLAGS.save_dir, argv[1], 'validation'), lossControl)


    return output
if __name__ == "__main__":
    tf.app.run()
