import tensorflow as tf
from model import save_images
import os
import time
import numpy as np
from utils import *
from ops import *
from IPython import embed

def train(model, sess):
    d_optim, g_optim = model.build_model()
    coord, threads, merged_sum = init_training(model, sess)
    start_time = time.time()
    print_time = time.time()

    dataset = load_dataset(model)
    N = dataset.num_examples
    max_iter = int(N/model.batch_size) * model.config.epoch

    print "[*] Traing Start : N=%d, Batch=%d, epoch=%d, max_iter=%d" \
        %(N, model.batch_size, model.config.epoch, max_iter)

    try:
        for idx in xrange(1, max_iter):
            batch_start_time = time.time()

            # D step
            image, label = dataset.next_batch(model.batch_size)
            # _, d_real, d_fake, summary = sess.run(
                # [d_optim, model.d_real, model.d_fake, merged_sum],
                # feed_dict={model.image:image, model.label:label})
            _, d_real, d_fake = sess.run(
                [d_optim, model.d_real, model.d_fake],
                feed_dict={model.image:image, model.label:label, model.z:get_z(model)})
            '''
            # Wasserstein
            _ = sess.run([model.clip_d_op])
            '''

            # G step
            image, label = dataset.next_batch(model.batch_size)
            # _, summary = sess.run([g_optim, merged_sum],
                                        # feed_dict={model.image:image, model.label:label})
            _ = sess.run([g_optim], feed_dict={model.image:image, model.label:label, model.z:get_z(model)})

            # save checkpoint for every epoch
            if (idx*model.batch_size) % N < model.batch_size:
                epoch = int(idx*model.batch_size/N)
                print_time = time.time()
                total_time = print_time - start_time
                sec_per_epoch = (print_time - start_time) / epoch

                summary = sess.run([merged_sum], feed_dict={model.image:image, model.label:label, model.z:get_z(model)})
                model.writer.add_summary(summary, idx)

                _save_samples(model, sess, epoch)
                model.save(sess, model.checkpoint_dir, epoch)

                print '[Epoch %(epoch)d] time: %(total_time)4.4f, d_real: %(d_real).8f, d_fake: %(d_fake).8f, sec_per_epoch: %(sec_per_epoch)4.4f' % locals()

    except tf.errors.OutOfRangeError:
        print "Done training; epoch limit reached."
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

def generate_grid_images(model, sess):
    d_optim, g_optim = model.build_model()
    coord, threads, merged_sum = init_training(model, sess)
    grid_size = 100
    samples = np.zeros(grid_size, grid_size, grid_size, 28, 28, 1)

    for i, y in enumerate(np.arange(-1., 1., 1./grid_size)):
        for j, x in enumerate(np.arange(1., -1., 1./grid_size)):
            z = get_z(model)
            z[:,0] = y
            z[:,1] = x
            samples[i, j, :] = sess.run([model.gen_image], feed_dict={model.z:z})

    with open(os.path.join(model.sample_dir, 'samples_grid_%d.npy'%(epoch)), 'w') as f:
        np.save(f, samples)

    coord.join(threads)
    sess.close()

def _save_samples(model, sess, epoch):
    samples = []
    noises = []

    # generator hard codes the batch size
    for i in xrange(model.sample_size // model.batch_size):
        # gen_image, noise = sess.run([model.gen_image, model.z])
        gen_image, noise = sess.run([model.gen_image, model.z],
                                    feed_dict={model.z:get_z(model)})
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

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    if model.load(sess, model.checkpoint_dir):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")

    if not os.path.exists(config.dataset_path):
        print(" [!] Data does not exist : %s" % config.dataset_path)
    return coord, threads, merged_sum

def load_dataset(model):
    if model.dataset_name == 'mnist':
        import mnist
        return mnist.read_data_sets(model.dataset_path, dtype=tf.uint8, reshape=False, validation_size=0).train
    elif model.dataset_name == 'cifar10':
        import cifar10
        return cifar10.read_data_sets(model.dataset_path, dtype=tf.uint8, reshape=False, validation_size=0).train

def get_z(model):
    return np.random.uniform(-1., 1., size=(model.batch_size, model.z_dim))
