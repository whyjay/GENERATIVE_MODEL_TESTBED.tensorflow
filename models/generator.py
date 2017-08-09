import tensorflow as tf
from ops import *
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

            h = fc(z, fc_dim)
            h = fc(h, f_dim*2*w/4*w/4)
            h = tf.reshape(h, [-1, w/4, w/4, f_dim*2])
            h = deconv2d(h, f_dim*2, 4, 2)
            x = deconv2d(h, c_dim, 4, 2, act=tf.nn.sigmoid, norm=None)

        else:
            n_layer = 4
            c = 2**(n_layer - 1)
            w = model.image_shape[0]/2**(n_layer)

            h = fc(z, f_dim * c * w * w)
            h = tf.reshape(h, [-1, w, w, f_dim * c])

            for i in range(n_layer - 1):
                w *= 2
                c /= 2
                h = deconv2d(h, gf_dim * c, 4, 2)

            x = deconv2d(h, c_dim, 4, 2, act=tf.nn.tanh, norm=None)

    return x

