# Code derived from tensorflow/tensorflow/models/image/imagenet/classify_image.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import tarfile

import sys
sys.path.insert(0, '/data/whyjay/NIPS2017')
import numpy as np
from six.moves import urllib
import tensorflow as tf
import glob
import scipy.misc
import math
import sys
from ops import *

MODEL_DIR = 'checkpoints'
softmax = None
x = None

# Call this function with list of images. Each of elements should be a
# numpy array with values ranging from 0 to 255.
def get_inception_score(images, splits=10):
    assert(type(images) == list)
    assert(type(images[0]) == np.ndarray)
    assert(len(images[0].shape) == 3)
    assert(np.min(images[0]) >= 0.0)
    inps = []

    for img in images:
        img = img.astype(np.float32)
        inps.append(np.expand_dims(img, 0))

    bs = 100
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(MODEL_DIR, ckpt_name))

        preds = []
        n_batches = int(math.ceil(float(len(inps)) / float(bs)))

        for i in range(n_batches):
            sys.stdout.write(".")
            sys.stdout.flush()
            inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
            inp = np.concatenate(inp, 0)
            pred = sess.run(softmax, {x: inp})
            preds.append(pred)

        preds = np.concatenate(preds, 0)
        scores = []

    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))

    return np.mean(scores), np.std(scores), preds

# This function is called automatically.
def _init_model():
    global softmax
    global x
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Works with an arbitrary minibatch size.
    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])

    h = slim.conv2d(images, 32, 3, 1, activation_fn=tf.nn.relu, normalizer_fn=None)
    h = tf.nn.max_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    h = slim.conv2d(images, 64, 3, 1, activation_fn=tf.nn.relu, normalizer_fn=None)
    h = tf.nn.max_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    h = tf.reshape(h, [-1, 7*7*64])
    h = slim.fully_connected(h, 1024, activation_fn=None, normalizer_fn=None)
    h = tf.nn.dropout(h, 1.0)
    logits = slim.fully_connected(h, 10, activation_fn=None, normalizer_fn=None)
    softmax = tf.nn.softmax(logits)

if softmax is None or x is None:
    _init_model()
