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

train_images_name = ['train_x_1.npy','train_x_2.npy','train_x_3.npy','train_x_4.npy']
train_labels_label = ['train_y_1.npy','train_y_2.npy','train_y_3.npy','train_y_4.npy']

test_images_name = ['test_x_1.npy','test_x_2.npy','test_x_3.npy','test_x_4.npy']
test_images_label = ['test_y_1.npy','test_y_2.npy','test_y_3.npy','test_y_4.npy']


def main(argv):
    # load data
    Best_set = []
    for i in range(4):
        train_images = np.load(FLAGS.data_dir + train_images_name[i])
        train_labels = np.load(FLAGS.data_dir + train_labels_label[i])
        test_images = np.load(FLAGS.data_dir + test_images_name[i])
        test_labels = np.load(FLAGS.data_dir + test_images_label[i])

    # train_images, validation_images,

        test_set_num_examples =  test_images.shape[0]
        train_num_examples = train_images.shape[0]
        train_images = np.reshape(train_images,[-1,129,129,1])
        test_images = np.reshape(test_images,[-1,129,129,1])
    #test_num_examples = test_images.shape[0]

    # specify the network
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
            for epoch in range(FLAGS.max_epoch_num):
                print('Epoch: ' + str(epoch))
                # run gradient steps and report mean loss on train data
                ce_vals = []
                for i in range(train_num_examples // batch_size):
                    batch_xs = train_images[i*batch_size:(i+1)*batch_size, :]
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
                lossControl.append (avg_test_cev)
                if epoch > 7 :
                    if (np.average(lossControl)+0.5*np.std(lossControl) < avg_test_cev):
                        print('Early stopping happens at ' + str(epoch))
                        print('the average+1std is: '+str(( np.average(lossControl)+np.std(lossControl))))
                        #path_prefix = saver.save(
                        #   session,
                        #   os.path.join(FLAGS.save_dir, argv[1], 'homework_1'),
                        #   global_step=global_step_tensor
                        #)
                        #saver = tf.train.import_meta_graph(path_prefix + '.meta')
                        break
                print('VALIDATION CONFUSION MATRIX:')
                confusion_sum = sum(conf_mxs_v)
                print(str(confusion_sum))

            #np.save(os.path.join(FLAGS.save_dir, argv[1], 'conf-matrix'), confusion_sum)
            #np.save(os.path.join(FLAGS.save_dir, argv[1], 'train'),
            #       training_avgs)
            #np.save(os.path.join(FLAGS.save_dir, argv[1], 'validation'), lossControl)

        Best_set[i] =  avg_test_cev

    Index = np.argmin (Best_set)

    train_images = np.load(FLAGS.data_dir + train_images_name[Index])
    train_labels = np.load(FLAGS.data_dir + train_labels_label[Index])
    test_images = np.load(FLAGS.data_dir + test_images_name[Index])
    test_labels = np.load(FLAGS.data_dir + test_images_label[Index])
    test_set_num_examples =  test_images.shape[0]
    train_num_examples = train_images.shape[0]
    train_images = np.reshape(train_images,[-1,129,129,1])
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
        for epoch in range(FLAGS.max_epoch_num):
            print('Epoch: ' + str(epoch))
            # run gradient steps and report mean loss on train data
            ce_vals = []
            for i in range(train_num_examples // batch_size):
                batch_xs = train_images[i*batch_size:(i+1)*batch_size, :]
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
            lossControl.append (avg_test_cev)
            if epoch > 7 :
                if (np.average(lossControl)+0.5*np.std(lossControl) < avg_test_cev):
                    print('Early stopping happens at ' + str(epoch))
                    print('the average+1std is: '+str(( np.average(lossControl)+np.std(lossControl))))
                    path_prefix = saver.save(
                        session,
                        os.path.join(FLAGS.save_dir, argv[1], 'homework_1'),
                        global_step=global_step_tensor
                    )
                    saver = tf.train.import_meta_graph(path_prefix + '.meta')
                    break
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
