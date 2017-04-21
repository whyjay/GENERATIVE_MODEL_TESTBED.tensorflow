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

        self.train = Train(self, config.train_func)
        #self.evaluate = Evaluate(self, config.eval_func)

        self.build_model = BuildModel(self, config.build_model_func)

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


class BuildModel(object):
    def __init__(self, model, func):
        self.model = model
        self.func = func

    def __call__(self):
        return self.func(self.model)

class NetworkWrapper(object):
    def __init__(self, model, func):
        self.model = model
        self.func = func

    def __call__(self, z, reuse=False):
        return self.func(self.model, z, reuse=reuse)


class Train(object):
    def __init__(self, model, func):
        self.model = model
        self.func = func

    def __call__(self, sess):
        return self.func(self.model, sess)
