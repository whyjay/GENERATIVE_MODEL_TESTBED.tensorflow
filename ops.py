import math
from contextlib import contextmanager
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

slim = tf.contrib.slim
rng = np.random.RandomState([2016, 6, 1])

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def sin_and_cos(x, name="ignored"):
    return tf.concat(len(x.get_shape()) - 1, [tf.sin(x), tf.cos(x)])

def maxout(x, k = 2):
    shape = [int(e) for e in x.get_shape()]
    ax = len(shape)
    ch = shape[-1]
    assert ch % k == 0
    shape[-1] = ch / k
    shape.append(k)
    x = tf.reshape(x, shape)
    return tf.reduce_max(x, ax)

def offset_maxout(x, k = 2):
    shape = [int(e) for e in x.get_shape()]
    ax = len(shape)
    ch = shape[-1]
    assert ch % k == 0
    shape[-1] = ch / k
    shape.append(k)
    x = tf.reshape(x, shape)
    ofs = rng.randn(1000, k).max(axis=1).mean()
    return tf.reduce_max(x, ax) - ofs

def lrelu_sq(x):
    """
    Concatenates lrelu and square
    """
    dim = len(x.get_shape()) - 1
    return tf.concat([lrelu(x), tf.minimum(tf.abs(x), tf.square(x))], dim)


def nin(input_, output_size, name=None, mean=0., stddev=0.02, bias_start=0.0, with_w=False):
    s = list(map(int, input_.get_shape()))
    input_ = tf.reshape(input_, [np.prod(s[:-1]), s[-1]])
    input_ = linear(input_, output_size, name=name, mean=mean, stddev=stddev, bias_start=bias_start, with_w=with_w)
    return tf.reshape(input_, s[:-1]+[output_size])

@contextmanager
def variables_on_cpu():
    old_fn = tf.get_variable
    def new_fn(*args, **kwargs):
        with tf.device("/cpu:0"):
            return old_fn(*args, **kwargs)
    tf.get_variable = new_fn
    yield
    tf.get_variable = old_fn

@contextmanager
def variables_on_gpu0():
    old_fn = tf.get_variable
    def new_fn(*args, **kwargs):
        with tf.device("/gpu:0"):
            return old_fn(*args, **kwargs)
    tf.get_variable = new_fn
    yield
    tf.get_variable = old_fn

def avg_grads(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.reduce_mean(tf.concat(grads, 0), 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def decayer(x, name="decayer"):
    with tf.variable_scope(name):
        scale = tf.get_variable("scale", [1], initializer=tf.constant_initializer(1.))
        decay_scale = tf.get_variable("decay_scale", [1], initializer=tf.constant_initializer(1.))
        relu = tf.nn.relu(x)
        return scale * relu / (1. + tf.abs(decay_scale) * tf.square(decay_scale))

def decayer2(x, name="decayer"):
    with tf.variable_scope(name):
        scale = tf.get_variable("scale", [int(x.get_shape()[-1])], initializer=tf.constant_initializer(1.))
        decay_scale = tf.get_variable("decay_scale", [int(x.get_shape()[-1])], initializer=tf.constant_initializer(1.))
        relu = tf.nn.relu(x)
        return scale * relu / (1. + tf.abs(decay_scale) * tf.square(decay_scale))

def masked_relu(x, name="ignored"):
    shape = [int(e) for e in x.get_shape()]
    prefix = [0] * (len(shape) - 1)
    most = shape[:-1]
    assert shape[-1] % 2 == 0
    half = shape[-1] // 2
    first_half = tf.slice(x, prefix + [0], most + [half])
    second_half = tf.slice(x, prefix + [half], most + [half])
    return tf.nn.relu(first_half) * tf.nn.sigmoid(second_half)

def make_z(shape, minval=-1.0, maxval=1.0, name="z"):
    z = tf.random_uniform(shape,
                        minval=minval, maxval=maxval,
                        name=name, dtype=tf.float32)
    #z = tf.random_normal(shape, name=name, stddev=0.5, dtype=tf.float32)
    return z

def get_sample_zs(model):
    assert model.sample_size > model.batch_size
    assert model.sample_size % model.batch_size == 0
    if model.config.multigpu:
        batch_size = model.batch_size // len(model.devices)
    else:
        batch_size = model.batch_size

    steps = model.sample_size // batch_size
    assert steps > 0

    sample_zs = []
    for i in xrange(steps):
        cur_zs = model.sess.run(model.z)
        sample_zs.append(cur_zs)

    sample_zs = np.concatenate(sample_zs, axis=0)
    assert sample_zs.shape[0] == model.sample_size
    return sample_zs

def batch_to_grid(images, width=4):
    images = tf.squeeze(images[:width**2])
    images_list = tf.unstack(images, num=width**2, axis=0)
    conc = tf.concat(images_list, axis=1)
    sp = tf.split(conc, width, axis=1)
    grid = tf.expand_dims(tf.concat(sp, axis=0), axis=0)
    if len(grid.get_shape().as_list()) < 4:
        grid = tf.expand_dims(grid, axis=-1)

    return grid

def fc(x, out_dim, act=tf.nn.relu, norm=slim.batch_norm, init=tf.truncated_normal_initializer(stddev=0.02)):
    return slim.fully_connected(
        x, out_dim, activation_fn=act, normalizer_fn=norm, weights_initializer=init)

def deconv2d(x, out_dim, k=4, s=2, act=tf.nn.relu, norm=slim.batch_norm, init=tf.truncated_normal_initializer(stddev=0.02)):
    return slim.conv2d_transpose(x, out_dim, k, s, activation_fn=act, normalizer_fn=norm, weights_initializer=init)

def conv2d(x, out_dim, k=4, s=2, act=tf.nn.relu, norm=slim.batch_norm, init=tf.truncated_normal_initializer(stddev=0.02)):
    return slim.conv2d(x, out_dim, k, s, activation_fn=act, normalizer_fn=norm, weights_initializer=init)


def preprocess_image(image, dataset, use_augmentation=False):
    image = tf.divide(image, 255., name=None)
    if use_augmentation:
        image = tf.image.random_brightness(image, max_delta=16. / 255.)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.minimum(tf.maximum(image, 0.0), 1.0)

    if 'mnist' not in dataset:
        image = tf.subtract(tf.divide(image, 255./2, name=None), 1)

    return image

def conv_mean_pool(x, out_dim, k=3, act=tf.nn.relu, norm=slim.batch_norm, init=tf.truncated_normal_initializer(stddev=0.02)):
    h = conv2d(x, out_dim, k=k, s=1, act=act, norm=norm, init=init)
    return tf.add_n([h[:,::2,::2,:], h[:,1::2,::2,:], h[:,::2,1::2,:], h[:,1::2,1::2,:]]) / 4.

def resize_conv2d(x, out_dim, k=3, scale=2, act=tf.nn.relu, norm=slim.batch_norm, init=tf.truncated_normal_initializer(stddev=0.02)):
    h, w = x.get_shape().as_list()[1:3]
    h = tf.image.resize_nearest_neighbor(x, (h*scale, w*scale), method=1)
    return conv2d(h, out_dim, k=k, s=1, act=act, norm=norm, init=init)

def residual_block(x, resample=None, no_dropout=False, labels=None):
    c_dim = x.get_shape().as_list()[-1]

    if resample=='down':
        h = conv2d(x, c_dim, 3, 1, init=None)
        h = conv_mean_pool(h, c_dim, 3, act=None, norm=None, init=None)
        shortcut = conv_mean_pool(x, c_dim, 1, act=None, norm=None, init=None)
    elif resample=='up':
        h = resize_conv2d(x, c_dim, 3, init=None)
        h = conv2d(h, c_dim, 3, 1, act=None, norm=None, init=None)
        shortcut = resize_conv2d(x, c_dim, 1, act=None, norm=None, init=None)
    elif resample==None:
        h = conv2d(x, c_dim, 3, 1, init=None)
        h = conv2d(h, c_dim, 3, 1, act=None, norm=None, init=None)
        shortcut = x
    else:
        raise Exception('invalid resample value')

    return tf.nn.relu(slim.batch_norm(shortcut + h))
