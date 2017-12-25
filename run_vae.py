import os
import numpy as np
import tensorflow as tf

from vae_models.config import Config
from vae_models.model import VAE
from vae_models.train import train
from utils import pp, visualize, to_json

from IPython import embed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.app.flags
flags.DEFINE_integer("epoch", 1500, "Max epoch to train")
flags.DEFINE_string("exp", 1, "Experiment number")
flags.DEFINE_string("batch_size", 64, "Batch size")
flags.DEFINE_string("learning_rate", 0.0002, "Learning rate")
flags.DEFINE_string("load_cp_dir", '', "checkpoint path")
flags.DEFINE_string("dataset", "cifar10", "[mnist, fashion, affmnist, cifar10]")
flags.DEFINE_string("latent_distribution", "gaussian", "[gaussian, vmf]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("use_augmentation", True, "Normalization and random brightness/contrast")
flags.DEFINE_string("encoder", 'base_encoder', '')
flags.DEFINE_string("decoder", 'base_decoder', '')

FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    config = Config(FLAGS)
    config.print_config()
    config.make_dirs()

    config_proto = tf.ConfigProto(allow_soft_placement=FLAGS.is_train, log_device_placement=False)
    config_proto.gpu_options.allow_growth = True

    with tf.Session(config=config_proto) as sess:
        model = VAE(config)
        train(model, sess)

if __name__ == '__main__':
    tf.app.run()
