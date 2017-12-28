import os
import time
from glob import glob
import tensorflow as tf
from tensorflow.python.ops import parsing_ops

from ops import *
from utils import *
from IPython import embed

slim = tf.contrib.slim

class VAE(object):
    def __init__(self, config):
        self.devices = config.devices
        self.config = config

        self.encoder = NetworkWrapper(self, config.encoder_func)
        self.decoder = NetworkWrapper(self, config.decoder_func)

        #self.evaluate = Evaluate(self, config.eval_func)

        self.batch_size = config.batch_size
        self.sample_size = config.sample_size
        self.image_shape = config.image_shape
        self.sample_dir = config.sample_dir

        self.k = config.kappa
        self.latent_distribution = config.latent_distribution
        self.y_dim = config.y_dim
        self.c_dim = config.c_dim
        self.f_dim = config.f_dim
        self.fc_dim = config.fc_dim
        self.z_dim = config.z_dim
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        self.dataset_name = config.dataset
        self.dataset_path = config.dataset_path
        self.checkpoint_dir = config.checkpoint_dir

        self.use_augmentation = config.use_augmentation
        self.is_training = tf.Variable(True, name='is_training', trainable=False)

    def save(self, sess, checkpoint_dir, step):
        model_name = "VAE.model"
        model_dir = "%s_%s" % (self.batch_size, self.config.learning_rate)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, sess, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s" % (self.batch_size, self.config.learning_rate)
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
        self.t_vars = tf.trainable_variables()

    def build_model(self):
        config = self.config

        # input
        self.image = tf.placeholder(tf.float32, shape=[self.batch_size]+self.image_shape)
        self.label = tf.placeholder(tf.float32, shape=[self.batch_size])
        self.noise = tf.placeholder(tf.float32, shape=[self.batch_size, self.z_dim])
        image = preprocess_image(self.image, self.dataset_name, self.use_augmentation)

        #self.z = make_z(shape=[self.batch_size, self.z_dim])

        z_mu, z_logvar = self.encoder(image)
        z = reparameterize(z_mu, z_logvar, self.latent_distribution)
        recon_image = self.decoder(z)

        loss_elbo, loss_recon, loss_kl = self.get_loss(image, recon_image, z_mu, z_logvar)

        # optimizer
        self.get_vars()
        opt = tf.train.AdamOptimizer(config.learning_rate, beta1=self.beta1, beta2=self.beta2)
        train_op = slim.learning.create_train_op(loss_elbo, opt, variables_to_train=self.t_vars)

        # logging
        tf.summary.scalar("loss_elbo", loss_elbo)
        tf.summary.scalar("loss_recon", loss_recon)
        tf.summary.scalar("loss_kl", loss_kl)
        tf.summary.image("input_images", batch_to_grid(image))
        tf.summary.image("recon_images", batch_to_grid(recon_image))

        self.recon_image = recon_image
        self.input_image = image
        self.z = z
        self.gen_image = self.decoder(self.noise)

        self.loss_elbo = loss_elbo
        self.loss_recon = loss_recon
        self.loss_kl = loss_kl + (np.prod(image.get_shape().as_list())-1.)/2.*np.log(2.)
        self.saver = tf.train.Saver(max_to_keep=None)

        return train_op

    def get_loss(self, image, recon_image, z_mu, z_logvar, eps = 1e-10):
        if self.dataset_name == 'mnist':
            loss_recon = -tf.reduce_sum(
                image * tf.log(eps+recon_image) + (1-image) * tf.log(eps+1-recon_image), axis=1
            )
        else:
            loss_recon = tf.reduce_sum(2. * tf.square(image - recon_image), axis=[1,2,3])
            loss_recon = tf.reduce_mean(loss_recon) / np.prod(self.image_shape)

        if self.latent_distribution == 'gaussian':
            loss_kl = -0.5 * tf.reduce_sum(
                1 + z_logvar - tf.square(z_mu) - tf.exp(z_logvar), axis=1
            )
        elif self.latent_distribution == 'vmf':
            loss_kl = tf.cast(self.k, tf.float32)

        loss_recon = tf.reduce_mean(loss_recon)
        loss_kl = tf.reduce_mean(loss_kl)
        loss_elbo = tf.reduce_mean(loss_recon + loss_kl)
        return loss_elbo, loss_recon, loss_kl

class NetworkWrapper(object):
    def __init__(self, model, func):
        self.model = model
        self.func = func

    def __call__(self, z, reuse=False):
        return self.func(self.model, z, reuse=reuse)


