import tensorflow as tf
from ops import *
slim = tf.contrib.slim

from IPython import embed


def base_generator(model, z, reuse=False):
    if model.dataset_name == 'mnist':
        n_layer = 2
    else:
        n_layer = 4

    bs = model.batch_size
    w_start = model.image_shape[0]/2**(n_layer)
    c_start = model.c_dim * 2**(n_layer)

    with tf.variable_scope('g_') as scope:
        if reuse:
            scope.reuse_variables()

        h = slim.fully_connected(h, w_start*w_start*c_start, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm)
        h = tf.reshape(h, [-1, w_start, w_start, c_start])

        for i in range(1, n_layer):
            out_shape = [model.batch_size]+[w_start*2**i]*2+[c_start*2**i]
            h = slim.conv2d_transpose(h, c, 4, 2, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm)
            h = slim.conv2d(h, c, [3, 3], 1, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm)

        i += 1
        c = c_start*2**i
        h = slim.conv2d_transpose(h, c, 4, 2, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm)
        x = slim.conv2d(h, c, [3, 3], 1, activation_fn=tf.nn.tanh, normalizer_fn=slim.batch_norm)

    return x

