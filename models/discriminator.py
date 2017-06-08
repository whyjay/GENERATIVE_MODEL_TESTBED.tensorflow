import tensorflow as tf
import numpy as np
from ops import *
slim = tf.contrib.slim

from IPython import embed

def base_discriminator(model, x, reuse=False):
    if model.dataset_name == 'mnist':
        n_layer = 1
    else:
        n_layer = 3

    bs = model.batch_size

    with tf.variable_scope('d_') as scope:
        if reuse:
            scope.reuse_variables()

        h = x
        for i in range(n_layer):
            input_channel = h.get_shape().as_list()[-1]
            h = slim.conv2d(h, input_channel, 3, 1, activation_fn=lrelu, normalizer_fn=slim.batch_norm)
            h = slim.conv2d(h, input_channel*2, 4, 2, activation_fn=lrelu, normalizer_fn=slim.batch_norm)

        h = tf.reshape(h, [bs, -1])
        logits = tf.reshape(slim.fully_connected(h, 1, activation_fn=None), [-1])
        probs = tf.nn.sigmoid(logits)

    return probs, logits


