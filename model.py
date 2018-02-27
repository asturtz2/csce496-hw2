import tensorflow as tf
import numpy as np
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
        latten_dim = np.prod(pool_2.get_shape().as_list()[1:])
        flat_conv = tf.reshape(pool_2, [-1, latten_dim])
        dens1 = tf.layers.dense(flat_conv, 64, activation=tf.nn.relu)
        output = tf.layers.dense(dens1, 7)
    return output


def model_conv_3(x):
    with tf.name_scope('Conv_model') as scope:
        hidden_1 = tf.layers.conv2d(x, 16, 3, padding='same', activation=tf.nn.relu, name='hidden_1')
        pool_1 = tf.layers.max_pooling2d(hidden_1, 2, 2, padding='same')
        hidden_2 = tf.layers.conv2d(pool_1, 32, 3, padding='same', activation=tf.nn.relu, name='hidden_2')
        pool_2 = tf.layers.max_pooling2d(hidden_2, 2, 2, padding='same')
        hidden_3 = tf.layers.conv2d(pool_2, 64, 3, padding='same', activation=tf.nn.relu, name='hidden_3')
        pool_3 = tf.layers.max_pooling2d(hidden_3, 2, 2, padding='same')
        latten_dim = np.prod(pool_3.get_shape().as_list()[1:])
        flat_conv = tf.reshape(pool_3, [-1, latten_dim])
        dens1 = tf.layers.dense(flat_conv, 128, activation=tf.nn.relu)
        dens2 = tf.layers.dense(dens1, 64, activation=tf.nn.relu)
        output = tf.layers.dense(dens2, 7)
    return output
