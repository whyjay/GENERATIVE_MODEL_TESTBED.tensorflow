import tensorflow as tf
import numpy as np
from ops import *
slim = tf.contrib.slim

from IPython import embed

def dcgan_d(model, x, reuse=False):
    bs = model.batch_size
    f_dim = model.f_dim
    fc_dim = model.fc_dim
    c_dim = model.c_dim

    with tf.variable_scope('d_', reuse=reuse) as scope:

        if model.dataset_name == 'mnist':
            w = model.image_shape[0]
            h = slim.conv2d(x, f_dim, 3, 1, activation_fn=lrelu, normalizer_fn=None)
            h = slim.conv2d(h, f_dim*2, 3, 1, activation_fn=lrelu, normalizer_fn=slim.batch_norm)
            h = tf.reshape(h, [bs, -1])
            h = slim.fully_connected(h, fc_dim, activation_fn=lrelu, normalizer_fn=slim.batch_norm)

        else:
            n_layer = 4
            c = 1
            w = model.image_shape[0]/2**(n_layer)

            h = slim.conv2d(x, f_dim * c, 4, 2, activation_fn=lrelu, normalizer_fn=None)
            for i in range(n_layer - 1):
                w /= 2
                c *= 2
                h = slim.conv2d(h, f_dim * c, 4, 2, activation_fn=lrelu, normalizer_fn=slim.batch_norm)

            h = tf.reshape(h, [bs, -1])

        logits = slim.fully_connected(h, 1, activation_fn=None)
        probs = tf.nn.sigmoid(logits)
    return probs, logits




