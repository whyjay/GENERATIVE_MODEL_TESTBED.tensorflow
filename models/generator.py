import tensorflow as tf
from ops import *

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

        h = tf.nn.relu(bn(linear(z, w_start*w_start*c_start, 'h0_lin', stddev=0.05), 'bn_0_lin'))
        h = tf.reshape(h, [-1, w_start, w_start, c_start])

        for i in range(1, n_layer):
            out_shape = [model.batch_size]+[w_start*2**i]*2+[c_start*2**i]
            h = tf.nn.relu(bn(deconv2d(h, out_shape, stddev=0.05, name='h%d'%i), 'bn%d'%i))
            h = tf.nn.relu(bn(conv2d(h, out_shape[-1], k=3, d=1, stddev=0.02, name='h%d_'%i), 'bn%d_'%i))

        i += 1
        out_shape = [model.batch_size]+[w_start*2**i]*2+[c_start*2**i]
        h = tf.nn.relu(bn(deconv2d(h, out_shape, stddev=0.05, name='h%d'%i), 'bn%d'%i))
        x = tf.nn.tanh(bn(conv2d(h, model.c_dim, k=3, d=1, stddev=0.02, name='h%d_'%i), 'bn%d_'%i))

    return x

