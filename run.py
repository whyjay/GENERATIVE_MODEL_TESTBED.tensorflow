import os
import numpy as np
import tensorflow as tf

from models.config import Config
from models.model import GAN
from models.train import train
from utils import pp, visualize, to_json

from IPython import embed

flags = tf.app.flags
flags.DEFINE_integer("epoch", 1000000, "Max epoch to train")
flags.DEFINE_string("exp", 0, "Experiment number")
flags.DEFINE_string("load_cp_dir", '', "cp path")
flags.DEFINE_string("dataset", "mnist", "[mnist, cifar10]")
flags.DEFINE_string("loss", "jsd", "[mnist, cifar10]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_string("generator", 'dcgan_g', '')
flags.DEFINE_string("discriminator", 'dcgan_d', '')

FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    config = Config(FLAGS)
    config.print_config()
    config.make_dirs()

    config_proto = tf.ConfigProto(allow_soft_placement=FLAGS.is_train, log_device_placement=False)
    config_proto.gpu_options.allow_growth = True

    with tf.Session(config=config_proto) as sess:
        model = GAN(config)
        if FLAGS.load_cp_dir is not '':
            model.load(FLAGS.load_cp_dir)

        train(model, sess)

        #OPTION = 2
        #visualize(sess, gan, FLAGS, OPTION)

if __name__ == '__main__':
    tf.app.run()
