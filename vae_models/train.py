import tensorflow as tf
import os
import time
import numpy as np
from utils import *
from ops import *
from IPython import embed

def train(model, sess):
    config = model.config
    train_op = model.build_model()

    if not (config.load_cp_dir == ''):
        model.load(sess, config.load_cp_dir)
    merged_sum = init_training(model, sess)
    start_time = time.time()
    print_time = time.time()

    dataset = load_dataset(model)
    N = dataset.num_examples
    max_iter = int(N/model.batch_size) * model.config.epoch

    print "[*] Traing Start : N=%d, Batch=%d, epoch=%d, max_iter=%d" \
        %(N, model.batch_size, model.config.epoch, max_iter)

    for idx in xrange(1, max_iter):
        batch_start_time = time.time()
        sess.run(model.is_training.assign(True))

        image, label = dataset.next_batch(model.batch_size)
        _, recon_image, input_image, z, loss_elbo, loss_kl, loss_recon = sess.run(
            [train_op, model.recon_image, model.input_image, model.z, model.loss_elbo, model.loss_kl, model.loss_recon],
            feed_dict={model.image:image, model.label:label})

        # save checkpoint for every epoch
        if (idx*model.batch_size) % N < model.batch_size:
            epoch = int(idx*model.batch_size/N)
            print_time = time.time()
            total_time = print_time - start_time
            sec_per_epoch = (print_time - start_time) / epoch

            image, label = dataset.next_batch(model.batch_size)
            summary = sess.run(merged_sum, feed_dict={model.image:image, model.label:label, model.z:get_z(model)})

            model.writer.add_summary(summary, idx)

            sess.run(model.is_training.assign(False))
            _save_samples(model, sess, epoch)
            model.save(sess, model.checkpoint_dir, epoch)

            print '[Epoch %(epoch)d] time: %(total_time)4.4f, loss_elbo: %(loss_elbo).4f, loss_kl: %(loss_kl).4f, loss_recon: %(loss_recon).4f, sec_per_epoch: %(sec_per_epoch)4.4f' % locals()

    sess.close()

def _save_samples(model, sess, epoch):
    samples = []
    noises = []

    # generator hard codes the batch size
    for i in xrange(model.sample_size // model.batch_size):
        noise = get_z(model)
        gen_image = sess.run(model.gen_image, feed_dict={model.noise:noise})
        samples.append(gen_image)
        noises.append(noise)

    samples = np.concatenate(samples, axis=0)
    noises = np.concatenate(noises, axis=0)

    assert samples.shape[0] == model.sample_size
    save_images(samples, [8, 8], os.path.join(model.sample_dir, 'samples_%s.png' % (epoch)))

    print  "Save Samples at %s/%s" % (model.sample_dir, 'samples_%s' % (epoch))
    with open(os.path.join(model.sample_dir, 'samples_%d.npy'%(epoch)), 'w') as f:
        np.save(f, samples)
    with open(os.path.join(model.sample_dir, 'noises_%d.npy'%(epoch)), 'w') as f:
        np.save(f, noises)

def init_training(model, sess):
    config = model.config
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    merged_sum = tf.summary.merge_all()
    model.writer = tf.summary.FileWriter(config.log_dir, sess.graph)

    if model.load(sess, model.checkpoint_dir):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")

    if not os.path.exists(config.dataset_path):
        print(" [!] Data does not exist : %s" % config.dataset_path)
    return merged_sum

def load_dataset(model):
    if model.dataset_name == 'mnist':
        import mnist as ds
    elif model.dataset_name == 'fashion':
        import fashion as ds
    elif model.dataset_name == 'cifar10':
        import cifar10 as ds
    return ds.read_data_sets(model.dataset_path, dtype=tf.uint8, reshape=False, validation_size=0).train

def get_z(model):
    if model.latent_distribution == 'vmf':
        z = np.random.normal(0., 1., size=(model.batch_size, model.z_dim))
        return z/np.linalg.norm(z)
    else:
        return np.random.normal(0., 1., size=(model.batch_size, model.z_dim))

