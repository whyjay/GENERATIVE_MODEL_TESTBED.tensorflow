import os
import time
from glob import glob
import tensorflow as tf
from tensorflow.python.ops import parsing_ops

from ops import *
from utils import *
from IPython import embed

class GAN(object):
    def __init__(self, config):
        self.devices = config.devices
        self.noise_stddev = config.noise_stddev
        self.config = config

        self.generator = NetworkWrapper(self, config.generator_func)
        self.discriminator = NetworkWrapper(self, config.discriminator_func)

        #self.evaluate = Evaluate(self, config.eval_func)

        self.batch_size = config.batch_size
        self.sample_size = config.sample_size
        self.image_shape = config.image_shape
        self.sample_dir = config.sample_dir

        self.y_dim = config.y_dim
        self.c_dim = config.c_dim
        self.f_dim = config.f_dim
        self.z_dim = config.z_dim

        self.dataset_name = config.dataset
        self.dataset_path = config.dataset_path
        self.checkpoint_dir = config.checkpoint_dir

    def save(self, sess, checkpoint_dir, step):
        model_name = "GAN.model"
        model_dir = "%s_%s_%s" % (self.batch_size, self.config.generator_learning_rate, self.config.discriminator_learning_rate)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, sess, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s_%s" % (self.batch_size, self.config.generator_learning_rate, self.config.discriminator_learning_rate)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            print "Bad checkpoint: ", ckpt
            return False

    def get_vars(self):
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if var.name.startswith('d_')]
        self.g_vars = [var for var in t_vars if var.name.startswith('g_')]

        for x in self.d_vars:
            assert x not in self.g_vars
        for x in self.g_vars:
            assert x not in self.d_vars
        for x in t_vars:
            assert x in  self.g_vars or x in self.d_vars, x.name
        self.all_vars = t_vars

    def build_model(self):
        config = self.config

        # input
        self.image = tf.placeholder(tf.float32, shape=[self.batch_size]+self.image_shape)
        self.label = tf.placeholder(tf.float32, shape=[self.batch_size])
        image = tf.subtract(tf.divide(self.image, 255./2, name=None), 1)

        self.z = make_z(shape=[self.batch_size, self.z_dim])
        self.gen_image = self.generator(self.z)
        d_out_real = self.discriminator(image)
        d_out_fake = self.discriminator(self.gen_image, reuse=True)

        d_loss, g_loss, d_real, d_fake = get_loss(d_out_real, d_out_fake, config.loss)

        # optimizer
        self.get_vars()
        d_opt = tf.train.RMSPropOptimizer(config.discriminator_learning_rate)
        g_opt = tf.train.RMSPropOptimizer(config.generator_learning_rate)
        d_grads = d_opt.compute_gradients(d_loss, var_list=self.d_vars)
        g_grads = g_opt.compute_gradients(g_loss, var_list=self.g_vars)
        d_optimize = d_opt.apply_gradients(d_grads)
        g_optimize = g_opt.apply_gradients(g_grads)

        # logging
        tf.summary.scalar("d_real", tf.reduce_mean(d_real))
        tf.summary.scalar("d_fake", tf.reduce_mean(d_fake))
        tf.summary.scalar("d_loss", d_loss)
        tf.summary.scalar("g_loss", g_loss)
        self.loss_real = loss_real
        self.loss_fake = loss_fake
        #tf.summary.histogram("d_grads", d_grads)
        #tf.summary.histogram("g_grads", g_grads)
        self.saver = tf.train.Saver(max_to_keep=None)

        return d_optimize, g_optimize

    def get_loss(d_out_real, d_out_fake, loss='jsd'):
        sigm_ce = tf.nn.sigmoid_cross_entropy_with_logits
        loss_real = - tf.reduce_mean(sigm_ce(logits=d_out_real, labels=tf.ones_like(d_out_real)))
        loss_fake = - tf.reduce_mean(sigm_ce(logits=d_out_fake, labels=tf.zeros_like(d_out_fake)))
        loss_fake_ = - tf.reduce_mean(sigm_ce(logits=d_out_fake, labels=tf.ones_like(d_out_fake)))

        if loss == 'jsd':
            d_loss = loss_real + loss_fake
            g_loss = - loss_fake
        elif loss == 'alternative':
            d_loss = loss_real + loss_fake
            g_loss = loss_fake_
        elif loss == 'reverse_kl':
            d_loss = loss_real + loss_fake
            g_loss = loss_fake_ - loss_fake

        return d_loss, g_loss, tf.nn.sigmoid(d_out_real), tf.nn.sigmoid(d_out_fake)



class NetworkWrapper(object):
    def __init__(self, model, func):
        self.model = model
        self.func = func

    def __call__(self, z, reuse=False):
        return self.func(self.model, z, reuse=reuse)


