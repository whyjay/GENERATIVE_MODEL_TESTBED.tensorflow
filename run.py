import os
import numpy as np
import tensorflow as tf

from models.config import Config
from models.model import GAN
from models.train import train
from utils import pp, visualize, to_json

from IPython import embed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.app.flags
flags.DEFINE_integer("epoch", 1000000, "Max epoch to train")
flags.DEFINE_string("exp", 0, "Experiment number")
flags.DEFINE_string("batch_size", 64, "Batch size")
flags.DEFINE_string("learning_rate", 0.0002, "Learning rate")
flags.DEFINE_string("load_cp_dir", '', "checkpoint path")
flags.DEFINE_string("dataset", "mnist", "[mnist, affmnist, cifar10]")
flags.DEFINE_string("loss", "jsd", "[jsd, alternative, reverse_kl]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("use_augmentation", True, "Normalization and random brightness/contrast")
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
        train(model, sess)
        #generate_grid_images(model, sess)
        #visualize(sess, gan, FLAGS, OPTION)

if __name__ == '__main__':
    tf.app.run()
