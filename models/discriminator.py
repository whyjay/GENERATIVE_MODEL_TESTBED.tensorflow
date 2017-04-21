import tensorflow as tf
import numpy as np
from ops import *

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
            h = lrelu(bn(conv2d(h, input_channel, k=3, d=1, stddev=0.02, name='h%d'%i), 'bn%d'%i))
            h = lrelu(bn(conv2d(h, input_channel*2, k=4, d=2, stddev=0.02, name='h%d_'%i), 'bn%d_'%i))

        h = tf.reshape(h, [bs, -1])
        logits = tf.reshape(linear(h, 1, name='d_logits'), [-1])
        probs = tf.nn.sigmoid(logits)

    return probs, logits


