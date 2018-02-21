import tensorflow as tf

def regularizer():
	return tf.contrib.layers.l2_regularizer(scale=1.0)

def dense_layer(x, layer_name, size, regularize = True):
	return tf.layers.dense(
		x,
		size,
		kernel_regularizer = regularizer() if regularize else None,
		bias_regularizer   = regularizer() if regularize else None,
		activation         = tf.nn.relu,
		name               = layer_name
	)
def my_conv_block(inputs, filters, is_training):
	"""
	Args:
		- inputs: 4D tensor of shape NHWC
		- filters: iterable of ints
	"""
	with tf.name_scope('conv_block') as scope:
		x = inputs
		for i in range(len(filters)):
			x = tf.layers.conv2d(x, filters[0], 3, 1, padding='same')
			x = tf.layers.batch_normalization(x, training=is_training)
			x = tf.nn.elu(x)
			x = tf.layers.max_pooling2d(x, 2, 2, padding='same')
	return x

def model_conv_2(x):
	with tf.name_scope('Conv_model') as scope:
		hidden_1 = tf.layers.conv2d(x, 64, 3, padding='same', activation=tf.nn.relu, name='hidden_1')
		pool_1 = tf.layers.max_pooling2d(hidden_1, 2, 2, padding='same')
		hidden_2 = tf.layers.conv2d(pool_1, 128, 3, padding='same', activation=tf.nn.relu, name='hidden_2')
		pool_2 = tf.layers.max_pooling2d(hidden_2, 2, 2, padding='same')
		flat_conv = tf.reshape(pool_2, [-1, 32*32*64])
		dens1 = tf.layers.dense(flat_conv, 128)
		output = tf.layers.dense(dens1, 7)
	return output


'''
# Rewrites
# def model_1(x):
#     with tf.name_scope('linear_model'):
#         hidden_1 = dense_layer(x, 'hidden_layer_1', 256, False)
#         hidden_2 = dense_layer(hidden_1, 'hidden_layer_2', 128, False)
#         return dense_layer(hidden_2, 'class_layer', 10, False)

# def model_2(x):
#     with tf.name_scope('linear_model'):
#         hidden_1 = dense_layer(x, 'hidden_layer_1', 512, False)
#         hidden_2 = dense_layer(hidden_1, 'hidden_layer_2', 256, False)
#         return dense_layer(hidden_2, 'class_layer', 10, False)

# def model_3(x):
#     with tf.name_scope('linear_model'):
#         hidden_1 = dense_layer(x, 'hidden_layer_1', 256)
#         hidden_2 = dense_layer(hidden_1, 'hidden_layer_2', 128)
#         return dense_layer(hidden_2, 'class_layer', 10)

# def model_4(x):
#     with tf.name_scope('linear_model'):
#         hidden_1 = dense_layer(x, 'hidden_layer_1', 512)
#         return dense_layer(hidden_1, 'class_layer', 10)

# def model_5(x):
#     keep_prob = 0.7
#     with tf.name_scope('linear_model'):
#         dropped_input = tf.layers.dropout(x, keep_prob)
#         hidden_1 = dense_layer(dropped_input, 'hidden_layer_1', 256, False)
#         dropped_hidden_1 = tf.layers.dropout(hidden_1, keep_prob)
#         hidden_2 = dense_layer(dropped_hidden_1, 'hidden_layer_2', 128, False)
#         dropped_hidden_2 = tf.layers.dropout(hidden_2, keep_prob)
#         return dense_layer(dropped_hidden_2, 'class_layer', 10, False)

# def model_6(x):
#     keep_prob = 0.7
#     with tf.name_scope('linear_model'):
#         dropped_input = tf.layers.dropout(x, keep_prob)
#         hidden_1 = dense_layer(dropped_input, 'hidden_layer_1', 512, False)
#         dropped_hidden_1 = tf.layers.dropout(hidden_1, keep_prob)
#         hidden_2 = dense_layer(dropped_hidden_1, 'hidden_layer_2', 256, False)
#         dropped_hidden_2 = tf.layers.dropout(hidden_2, keep_prob)
#         return dense_layer(dropped_hidden_2, 'class_layer', 10, False)

# def model_7(x):
#     keep_prob = 0.7
#     with tf.name_scope('linear_model'):
#         dropped_input = tf.layers.dropout(x, keep_prob)
#         hidden_1 = dense_layer(dropped_input, 'hidden_layer_1', 256)
#         dropped_hidden_1 = tf.layers.dropout(hidden_1, keep_prob)
#         hidden_2 = dense_layer(dropped_hidden_1, 'hidden_layer_2', 128)
#         dropped_hidden_2 = tf.layers.dropout(hidden_2, keep_prob)
#         return dense_layer(dropped_hidden_2, 'class_layer', 10)

# def model_8(x):
#     keep_prob = 0.7
#     with tf.name_scope('linear_model'):
#         dropped_input = tf.layers.dropout(x, keep_prob)
#         hidden_1 = dense_layer(dropped_input, 'hidden_layer_1', 512)
#         dropped_hidden_1 = tf.layers.dropout(hidden_1, keep_prob)
#         hidden_2 = dense_layer(dropped_hidden_1, 'hidden_layer_2', 256)
#         dropped_hidden_2 = tf.layers.dropout(hidden_2, keep_prob)
#         return dense_layer(dropped_hidden_2, 'class_layer', 10)


def model_1(x):
	x = x / 255.0
	with tf.name_scope('linear_model') as scope:
		hidden_1 = tf.layers.dense(
			x,
			256,
			activation=tf.nn.relu,
			name='hidden_layer_1')
		hidden_2 = tf.layers.dense(
			hidden_1,
			128,
			activation=tf.nn.relu,
			name='hidden_layer_2')
		output = tf.layers.dense(
			hidden_2,
			10,
			name='class_layer')
		return output

def model_2(x):
	with tf.name_scope('linear_model') as scope:
		hidden_1 = tf.layers.dense(
			x,
			512,
			activation=tf.nn.relu,
			name='hidden_layer_1')
		hidden_2 = tf.layers.dense(
			hidden_1,
			256,
			activation=tf.nn.relu,
			name='hidden_layer_2')
		output = tf.layers.dense(
			hidden_2,
			10,
			name='class_layer')
		return output

def model_3(x):
	x = x / 255.0
	with tf.name_scope('linear_model') as scope:
		hidden_1 = tf.layers.dense(
			x,
			256,
			kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
			bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
			activation=tf.nn.relu,
			name='hidden_layer_1')
		hidden_2 = tf.layers.dense(
			hidden_1,
			128,
			kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
			bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
			activation=tf.nn.relu,
			name='hidden_layer_2')
		output = tf.layers.dense(
			hidden_2,
			10,
			kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
			bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
			name='class_layer')
		return output

def model_4(x):
	x = x / 255.0
	with tf.name_scope('linear_model') as scope:
		hidden_1 = tf.layers.dense(
			x,
			512,
			kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
			bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
			activation=tf.nn.relu,
			name='hidden_layer_1')
		hidden_2 = tf.layers.dense(
			hidden_1,
			256,
			kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
			bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
			activation=tf.nn.relu,
			name='hidden_layer_2')
		output = tf.layers.dense(
			hidden_2,
			10,
			kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
			bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
			name='class_layer')
		return output

def model_5(x):
	x = x / 255.0
	KEEP_PROB=0.7
	with tf.name_scope('linear_model') as scope:
		dropped_input = tf.layers.dropout(x, KEEP_PROB)
		hidden_1 = tf.layers.dense(
			dropped_input,
			256,
			activation=tf.nn.relu,
			name='hidden_layer_1')
		dropped_hidden_1 = tf.layers.dropout(hidden_1, KEEP_PROB)
		hidden_2 = tf.layers.dense(
			dropped_hidden_1,
			128,
			activation=tf.nn.relu,
			name='hidden_layer_2')
		dropped_hidden_2 = tf.layers.dropout(hidden_2, KEEP_PROB)
		output = tf.layers.dense(
			dropped_hidden_2,
			10,
			name='output_layer')
		return output

def model_6(x):
	x = x / 255.0
	KEEP_PROB=0.7
	with tf.name_scope('linear_model') as scope:
		dropped_input = tf.layers.dropout(x, KEEP_PROB)
		hidden_1 = tf.layers.dense(
			dropped_input,
			512,
			activation=tf.nn.relu,
			name='hidden_layer_1')
		dropped_hidden_1 = tf.layers.dropout(hidden_1, KEEP_PROB)
		hidden_2 = tf.layers.dense(
			dropped_hidden_1,
			256,
			activation=tf.nn.relu,
			name='hidden_layer_2')
		dropped_hidden_2 = tf.layers.dropout(hidden_2, KEEP_PROB)
		output = tf.layers.dense(
			dropped_hidden_2,
			10,
			name='output_layer')
		return output

def model_7(x):
	x = x / 255.0
	KEEP_PROB=0.7
	with tf.name_scope('linear_model') as scope:
		dropped_input = tf.layers.dropout(x, KEEP_PROB)
		hidden_1 = tf.layers.dense(
			dropped_input,
			256,
			kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
			bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
			activation=tf.nn.relu,
			name='hidden_layer_1')
		dropped_hidden_1 = tf.layers.dropout(hidden_1, KEEP_PROB)
		hidden_2 = tf.layers.dense(
			dropped_hidden_1,
			128,
			kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
			bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
			activation=tf.nn.relu,
			name='hidden_layer_2')
		dropped_hidden_2 = tf.layers.dropout(hidden_2, KEEP_PROB)
		output = tf.layers.dense(
			dropped_hidden_2,
			10,
			kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
			bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
			name='output_layer')
		return output

def model_8(x):
	x = x / 255.0
	KEEP_PROB=0.7
	with tf.name_scope('linear_model') as scope:
		dropped_input = tf.layers.dropout(x, KEEP_PROB)
		hidden_1 = tf.layers.dense(
			dropped_input,
			512,
			kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
			bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
			activation=tf.nn.relu,
			name='hidden_layer_1')
		dropped_hidden_1 = tf.layers.dropout(hidden_1, KEEP_PROB)
		hidden_2 = tf.layers.dense(
			dropped_hidden_1,
			256,
			kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
			bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
			activation=tf.nn.relu,
			name='hidden_layer_2')
		dropped_hidden_2 = tf.layers.dropout(hidden_2, KEEP_PROB)
		output = tf.layers.dense(
			dropped_hidden_2,
			10,
			kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
			bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
			name='output_layer')
		return output


# def model_4(x):
#     with tf.name_scope('linear_model') as scope:
#         hidden_1 = tf.layers.dense(x,
#              512,
#              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
#              bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
#              activation=tf.nn.relu,
#              name='hidden_layer')
#         hidden_2 = tf.layers.dense(
#              hidden_1,
#              512,
#              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
#              bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
#              activation=tf.nn.relu,
#              name='hidden_layer_2')
#         hidden_3 = tf.layers.dense(
#              hidden_2,
#              512,
#              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
#              bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
#              activation=tf.nn.relu,
#              name='hidden_layer_3')
#         output = tf.layers.dense(
#              hidden_3,
#              10,
#              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
#              bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
#              name='class_layer')
#         return output

# def model_5(x):
#     KEEP_PROB=0.7
#     with tf.name_scope('linear_model') as scope:

#         dropped_input = tf.layers.dropout(x, KEEP_PROB)
#         hidden = tf.layers.dense(dropped_input,
#                              512,
#                              activation=tf.nn.relu,
#                              name='hidden_layer')
#         dropped_hidden = tf.layers.dropout(hidden, KEEP_PROB)
#         output = tf.layers.dense(dropped_hidden,
#                              10,
#                              name='output_layer')
#         return output

# def deep_dropout_model(inputs, keep_prob):
#     with tf.name_scope('linear_model') as scope:
#         dropped_input = tf.layers.Dropout(keep_prob)
#         hidden_1 = dense_layer('hidden_layer_1', size = 256, regularize = False)
#         hidden_2 = dense_layer('hidden_layer_2', size = 256, regularize = False)
#         dropped_hidden = tf.layers.Dropout(keep_prob)
#         output = dense_layer('class_layer', size = 10, regularize = False)
#         return output(dropped_hidden(hidden_2(hidden_1(dropped_input(inputs)))))


# def model_6(x):
#     KEEP_PROB=0.8
#     with tf.name_scope('linear_model') as scope:

#         dropped_input = tf.layers.dropout(x, KEEP_PROB)
#         hidden = tf.layers.dense(dropped_input,
#                              256,
#                              activation=tf.nn.relu,
#                              name='hidden_layer')
#         hidden2 = tf.layers.dense(hidden,
#                              256,
#                              activation=tf.nn.relu,
#                              name='hidden_layer2')
#         dropped_hidden = tf.layers.dropout(hidden2, KEEP_PROB)
#         output = tf.layers.dense(dropped_hidden,
#                              10,
#                              name='output_layer')
#         return output
'''