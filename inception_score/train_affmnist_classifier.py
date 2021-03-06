import sys
sys.path.insert(0, '../')
import math

import numpy as np
from scipy.ndimage import rotate
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from ops import *
from IPython import embed
slim = tf.contrib.slim

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

def main():
    sess = tf.Session()
    mnist = input_data.read_data_sets('../dataset/mnist', one_hot=True)
    save_path = 'checkpoints'

    x = tf.placeholder(tf.float32, shape=[None, 40, 40, 1])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    h = slim.conv2d(x, 32, 3, 1, activation_fn=tf.nn.relu, normalizer_fn=None)
    h = tf.nn.max_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    h = slim.conv2d(h, 64, 3, 1, activation_fn=tf.nn.relu, normalizer_fn=None)
    h = tf.nn.max_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    h = tf.reshape(h, [-1, 10*10*64])
    h = slim.fully_connected(h, 1024, activation_fn=None, normalizer_fn=None)
    keep_prob = tf.placeholder(tf.float32)
    h = tf.nn.dropout(h, keep_prob)
    logits = slim.fully_connected(h, 10, activation_fn=None, normalizer_fn=None)

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(max_to_keep=None)

    test_images = mnist.test.images
    test_images = test_images.reshape((-1, 28, 28, 1))
    move = np.zeros((len(test_images), 40, 40, 1)) # initialize
    for i in range(len(test_images)):
        degree = np.random.randint(41) - 20
        h_move = np.random.randint(12)
        w_move = np.random.randint(12)
        rot = rotate(test_images[i], degree, reshape = False)
        move[i][h_move:h_move+28, w_move:w_move+28] = rot
    test_images = move

    for i in range(200000):
        batch = mnist.train.next_batch(50)
        batch_img = batch[0].reshape((-1, 28, 28, 1))
        move = np.zeros((50, 40, 40, 1)) # initialize
        for j in range(50):
            degree = np.random.randint(41) - 20
            h_move = np.random.randint(12)
            w_move = np.random.randint(12)
            rot = rotate(batch_img[j], degree, reshape = False)
            move[j][h_move:h_move+28, w_move:w_move+28] = rot
        batch_img = move

        sess.run(train_step, feed_dict={x: batch_img, y_: batch[1], keep_prob: 0.5})

        if i % 1000 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x:batch_img, y_: batch[1], keep_prob: 1.0})
            test_accuracy = sess.run(accuracy, feed_dict={x:test_images, y_: mnist.test.labels, keep_prob: 1.0})
            save_at = os.path.join(save_path, 'affmnist_ckpt_%f' % test_accuracy)

            print("step %d, training accuracy %g, test accuracy %f"%(i, train_accuracy, test_accuracy))

            print 'Save at %s' % save_at
            saver.save(sess, save_at, global_step=i)

if __name__ == '__main__':
    main()
