import tensorflow as tf
import numpy as np
from ops import *
from IPython import embed

def dcgan_d(model, x, reuse=False):
    bs = model.batch_size
    f_dim = model.f_dim
    fc_dim = model.fc_dim
    c_dim = model.c_dim

    with slim.arg_scope(ops_with_bn, is_training=model.is_training):
        with tf.variable_scope('d_', reuse=reuse) as scope:

            if model.dataset_name == 'mnist':
                w = model.image_shape[0]
                h = conv2d(x, f_dim, 3, 1, act=lrelu, norm=None)
                h = conv2d(h, f_dim*2, 3, 1, act=lrelu)
                h = tf.reshape(h, [bs, -1])
                h = fc(h, fc_dim, act=lrelu)

            elif model.dataset_name == 'affmnist':
                n_layer = 3
                c = 1
                w = model.image_shape[0]/2**(n_layer)

                h = conv2d(x, f_dim * c, 4, 2, act=lrelu, norm=None)
                for i in range(n_layer - 1):
                    w /= 2
                    c *= 2
                    h = conv2d(h, f_dim * c, 4, 2, act=lrelu)
                    h = conv2d(h, f_dim * c, 1, 1, act=lrelu)

                    if i == n_layer - 2:
                        feats = h

            elif model.dataset_name == 'cifar10':
                h = conv2d(x, f_dim, 3, 1, act=tf.nn.elu, norm=None)
                h = conv_mean_pool(h, f_dim, 3, act=None, norm=None)
                h += conv_mean_pool(x, f_dim, 1, act=None, norm=None)
                h = tf.nn.elu(ln(h))

                h = residual_block(h, resample='down', act=tf.nn.elu)
                h = residual_block(h, resample=None, act=tf.nn.elu)
                h = residual_block(h, resample=None, act=tf.nn.elu)

                h = conv2d(h, f_dim, 4, 2, act=None, norm=ln)
                h = tf.reduce_mean(h, axis=[1,2])

            else:
                n_layer = 4
                c = 1
                w = model.image_shape[0]/2**(n_layer)

                h = conv2d(x, f_dim * c, 4, 2, act=lrelu, norm=None)
                for i in range(n_layer - 1):
                    w /= 2
                    c *= 2
                    h = conv2d(h, f_dim * c, 4, 2, act=lrelu)
                    h = conv2d(h, f_dim * c, 1, 1, act=lrelu)

                    if i == n_layer - 2:
                        feats = h

            h = tf.reshape(h, [bs, -1])
            logits = fc(h, 1, act=None, norm=None)
        return logits
