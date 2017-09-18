import os
import time
import datetime
from glob import glob
import tensorflow as tf

from ops import *
from utils import *

from models.generator import *
from models.discriminator import *
#from models.evaluate import evaluate
from utils import pp, visualize, to_json

from IPython import embed

class Config(object):
    def __init__(self, FLAGS):
        self.exp_num = str(FLAGS.exp)
        self.dataset = FLAGS.dataset
        self.dataset_path = os.path.join("./dataset/", self.dataset)
        self.devices = ["gpu:0", "gpu:1", "gpu:2", "gpu:3"]
        self.use_augmentation = FLAGS.use_augmentation
        self.batch_size = FLAGS.batch_size
        self.learning_rate = FLAGS.learning_rate

        self.add_noise = True
        self.noise_stddev = 0.1


        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")+str(self.learning_rate)

        self.epoch = FLAGS.epoch
        self.log_dir = os.path.join('logs', self.exp_num, timestamp)
        self.checkpoint_dir = os.path.join('checkpoint', self.exp_num, timestamp)
        self.sample_dir = os.path.join('samples', self.exp_num, timestamp)
        self.timestamp = timestamp

        self.generator_name = FLAGS.generator
        self.discriminator_name = FLAGS.discriminator

        self.generator_func = globals()[self.generator_name]
        self.discriminator_func = globals()[self.discriminator_name]

        self.loss = FLAGS.loss



        if FLAGS.dataset == "mnist":
            self.y_dim=10
            self.image_shape=[28, 28, 1]
            self.c_dim=1
            self.z_dim=100
            self.f_dim = 64
            self.fc_dim = 1024
            self.beta1 = 0.5
            self.beta2 = 0.999

        elif FLAGS.dataset == "affmnist":
            self.y_dim=10
            self.image_shape=[40, 40, 1]
            self.c_dim=1
            self.z_dim=128
            self.f_dim = 64
            self.fc_dim = 1024
            self.beta1 = 0.5
            self.beta2 = 0.999

        elif FLAGS.dataset == "celebA":
            self.y_dim=1
            self.image_shape=[64, 64, 3]
            self.c_dim=3
            self.z_dim=256 # 256, 10
            self.f_dim = 64
            self.fc_dim = 1024
            self.beta1 = 0.5
            self.beta2 = 0.999

        elif FLAGS.dataset == "cifar10":
            self.y_dim=10
            self.image_shape=[32, 32, 3]
            self.c_dim=3
            self.z_dim=128
            self.f_dim = 128
            self.fc_dim = 1024
            self.beta1 = 0.
            self.beta2 = 0.9

        self.sample_size=10*self.batch_size

    def print_config(self):
        dicts = self.__dict__
        for key in dicts.keys():
            print key, dicts[key]

    def make_dirs(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
