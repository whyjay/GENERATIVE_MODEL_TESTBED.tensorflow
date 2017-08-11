import tensorflow as tf
import numpy as np
from ops import *
from IPython import embed

def dcgan_d(model, x, reuse=False):
    bs = model.batch_size
    f_dim = model.f_dim
    fc_dim = model.fc_dim
    c_dim = model.c_dim

    with tf.variable_scope('d_', reuse=reuse) as scope:

        if model.dataset_name == 'mnist':
            w = model.image_shape[0]
            h = conv2d(x, f_dim, 3, 1, act=lrelu, norm=None)
            h = conv2d(h, f_dim*2, 3, 1, act=lrelu)
            h = tf.reshape(h, [bs, -1])
            h = fc(h, fc_dim, act=lrelu)

        else:
            n_layer = 4
            c = 1
            w = model.image_shape[0]/2**(n_layer)

            h = conv2d(x, f_dim * c, 4, 2, act=lrelu, norm=None)
            for i in range(n_layer - 1):
                w /= 2
                c *= 2
                h = conv2d(h, f_dim * c, 4, 2, act=lrelu)

            h = tf.reshape(h, [bs, -1])

        logits = fc(h, 1, act=None, norm=None)
    return logits
