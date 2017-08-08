import tensorflow as tf
from ops import *
slim = tf.contrib.slim

from IPython import embed

def dcgan_g(model, z, reuse=False):
    bs = model.batch_size
    f_dim = model.f_dim
    fc_dim = model.fc_dim
    c_dim = model.c_dim

    with tf.variable_scope('g_', reuse=reuse) as scope:

        if model.dataset_name == 'mnist':
            n_layer = 2
            w = model.image_shape[0]

            h = slim.fully_connected(z, fc_dim, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm)
            h = slim.fully_connected(h, f_dim*2*(w/4)*(w/4), activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm)
            h = tf.reshape(h, [-1, w/4, w/4, f_dim*2])
            h = slim.conv2d_transpose(h, f_dim*2, 4, 2, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm)
            x = slim.conv2d_transpose(h, c_dim, 4, 2, activation_fn=tf.nn.sigmoid, normalizer_fn=None)

        else:
            n_layer = 4
            c = 2**(n_layer - 1)
            w = model.image_shape[0]/2**(n_layer)

            h = slim.fully_connected(z, f_dim * c * w * w, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm)
            h = tf.reshape(h, [-1, w, w, f_dim * c])

            for i in range(n_layer - 1):
                w *= 2
                c /= 2
                h = slim.conv2d_transpose(h, gf_dim * c, 4, 2, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm)

            x = slim.conv2d_transpose(h, c_dim, 4, 2, activation_fn=tf.nn.tanh, normalizer_fn=None)

    return x

