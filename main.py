import tensorflow as tf
import numpy as np
import os
import util
import model

## This code is updated for homework #2
## Programmed by M.Ebrahim Mohammadi and Alex Strutz
## CSCE 896dl - Prof. Scott class 
## The main file is tuned to run EMODB training files
 

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/work/cse496dl/shared/homework/02/EMODB-German/', 'directory where EMODB is located')
#TODO: What to use as save dir?
flags.DEFINE_string('save_dir', '/home/structures/ebrahim31/homework2_output/', 'directory where model graph and weights are saved')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('proportion', 0.9, '')
flags.DEFINE_integer('max_epoch_num', 200, '')
FLAGS = flags.FLAGS



def main(argv):
	# load train data
	train_images_1 = np.load(FLAGS.data_dir + 'train_x_1.npy')
	train_labels_1 = np.load(FLAGS.data_dir + 'train_y_1.npy')
	train_images_2 = np.load(FLAGS.data_dir + 'train_x_2.npy')
	train_labels_2 = np.load(FLAGS.data_dir + 'train_y_2.npy')
	train_images_3 = np.load(FLAGS.data_dir + 'train_x_3.npy')
	train_labels_3 = np.load(FLAGS.data_dir + 'train_y_3.npy')
	train_images_4 = np.load(FLAGS.data_dir + 'train_x_4.npy')
	train_labels_4 = np.load(FLAGS.data_dir + 'train_y_4.npy')

	# load test data
	test_images_1 = np.load(FLAGS.data_dir + 'test_x_1.npy')
	test_labels_1 = np.load(FLAGS.data_dir + 'test_y_1.npy')
	test_images_2 = np.load(FLAGS.data_dir + 'test_x_2.npy')
	test_labels_2 = np.load(FLAGS.data_dir + 'test_y_2.npy')
	test_images_3 = np.load(FLAGS.data_dir + 'test_x_3.npy')
	test_labels_3 = np.load(FLAGS.data_dir + 'test_y_3.npy')
	test_images_4 = np.load(FLAGS.data_dir + 'test_x_4.npy')
	test_labels_4 = np.load(FLAGS.data_dir + 'test_y_4.npy')

	#######################################


	validation_set_num_examples =  validation_image.shape[0]
	train_num_examples = train_images_2.shape[0]
	#test_num_examples = test_images.shape[0]



	# TODO: Rewrite in terms of objects and with new architecture
	# specify the network
	input_placeholder = tf.placeholder(tf.float32, [None, 16641],
			name='input_placeholder')
	output = model.layers(input_placeholder)
	# define classification loss
	y = tf.placeholder(tf.float32, [None, 7], name='label')

	with tf.name_scope('cross_entropy') as scope:
		cross_entropy  = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)
		cross_entropy = tf.reduce_mean(cross_entropy)

	confusion_matrix_op = tf.confusion_matrix(tf.argmax(y, axis=1), tf.argmax(output, axis=1), num_classes=10)

	# set up training and saving functionality
	global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
	optimizer = tf.train.AdamOptimizer()
	train_op = optimizer.minimize(cross_entropy, global_step=global_step_tensor)
	saver = tf.train.Saver()

	with tf.Session() as session:
		session.run(tf.global_variables_initializer())

		# run training
		batch_size = FLAGS.batch_size
		lossControl = []
		for epoch in range(FLAGS.max_epoch_num):
			print('Epoch: ' + str(epoch))
			# run gradient steps and report mean loss on train data
			ce_vals = []

			for i in range(train_num_examples // batch_size):
				batch_xs = train_images_2[i*batch_size:(i+1)*batch_size, :]
				batch_ys = train_labels_2[i*batch_size:(i+1)*batch_size, :]
				batch_xs = batch_xs //255
				#batch_ys = batch_ys//255
				#_, train_ce = session.run([train_op, tf.reduce_mean(cross_entropy)], {x: batch_xs, y: batch_ys})
				_, train_ce = session.run([train_op, cross_entropy], {input_placeholder: batch_xs, y: batch_ys})
				ce_vals.append(train_ce)

			avg_train_ce = sum(ce_vals) / len(ce_vals)
		#   avg_test_cev = sum(ce_vals_v) / len(ce_vals_v)
		#   print('VALIDATION CROSS ENTROPY: ' + str(avg_test_cev))
		#   print('VALIDATION CONFUSION MATRIX:')
		#   print(str(sum(conf_mxs_v)))
			print('TRAIN CROSS ENTROPY: ' + str(avg_train_ce))

			# report mean test loss
#            ce_vals = []
#            conf_mxs = []
#            for i in range(test_num_examples // batch_size):
#                batch_xs = test_images[i*batch_size:(i+1)*batch_size, :]
#                batch_ys = test_labels[i*batch_size:(i+1)*batch_size, :]
#                test_ce, conf_matrix = session.run([cross_entropy, confusion_matrix_op], {x: batch_xs, y: batch_ys})
#                ce_vals.append(test_ce)
#                conf_mxs.append(conf_matrix)
#            avg_test_ce = sum(ce_vals) / len(ce_vals)
#            print('TEST CROSS ENTROPY: ' + str(avg_test_ce))
#            #print('TEST CONFUSION MATRIX:')
#            #print(str(sum(conf_mxs)))

			ce_vals_v = []
			conf_mxs_v = []
			for i in range(validation_set_num_examples // batch_size):
				batch_xsv = validation_image[i*batch_size:(i+1)*batch_size, :]
				batch_ysv = validation_labels[i*batch_size:(i+1)*batch_size, :]
				batch_xsv = batch_xsv//255
				#batch_ysv = batch_ysv //255
				test_cev, conf_matrix_v, _ = session.run([cross_entropy, confusion_matrix_op, output], {input_placeholder: batch_xsv, y: batch_ysv})
				ce_vals_v.append(test_cev)
				conf_mxs_v.append(conf_matrix_v)
			avg_test_cev = sum(ce_vals_v) / len(ce_vals_v)
			print('VALIDATION CROSS ENTROPY: ' + str(avg_test_cev))
			lossControl.append (avg_test_cev)
			if epoch > 5 : 
				if (( np.average(lossControl)+1.5*np.std(lossControl)) < avg_test_cev):
					print('Early stopping happens at ' + str(epoch))
					print('the average+1.5std is: '+str(( np.average(lossControl)+np.std(lossControl))))
					break


			#print('VALIDATION CONFUSION MATRIX:')
			#print(str(sum(conf_mxs_v)))

	#print(output)
	saver.save(session,  "homework_1", global_step=global_step_tensor)
	return print(output)
if __name__ == "__main__":
	tf.app.run()
