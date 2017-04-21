import tensorflow as tf
from ops import make_z

from IPython import embed

def build_model(model):
    config = model.config

    # input
    model.image = tf.placeholder(tf.float32, shape=[model.batch_size]+model.image_shape)
    model.label = tf.placeholder(tf.float32, shape=[model.batch_size])

    if 'mnist' in model.dataset_name:
        model.image = tf.divide(model.image, 255., name=None)
    else:
        model.image = tf.subtract(tf.divide(model.image, 255./2, name=None), 1)

    model.z = make_z(shape=[model.batch_size, model.z_dim])
    model.gen_image = model.generator(model.z)
    d_real, logits_real = model.discriminator(model.image)
    d_fake, logits_fake = model.discriminator(model.gen_image, reuse=True)

    # loss
    sigm_ce = tf.nn.sigmoid_cross_entropy_with_logits
    loss_real = - tf.reduce_mean(sigm_ce(logits=logits_real, labels=tf.ones_like(d_real)))
    loss_fake = - tf.reduce_mean(sigm_ce(logits=logits_fake, labels=tf.zeros_like(d_fake)))

    d_loss = loss_real + loss_fake
    g_loss = - loss_fake


    # optimizer
    model.get_vars()
    d_opt = tf.train.RMSPropOptimizer(config.discriminator_learning_rate)
    g_opt = tf.train.RMSPropOptimizer(config.generator_learning_rate)
    d_grads = d_opt.compute_gradients(d_loss, var_list=model.d_vars)
    g_grads = g_opt.compute_gradients(g_loss, var_list=model.g_vars)
    d_optimize = d_opt.apply_gradients(d_grads)
    g_optimize = g_opt.apply_gradients(g_grads)

    # logging
    tf.summary.scalar("d_real", tf.reduce_mean(d_real))
    tf.summary.scalar("d_fake", tf.reduce_mean(d_fake))
    tf.summary.scalar("logits_real", tf.reduce_mean(logits_real))
    tf.summary.scalar("logits_fake", tf.reduce_mean(logits_fake))
    tf.summary.scalar("d_loss", d_loss)
    tf.summary.scalar("g_loss", g_loss)
    model.loss_real = loss_real
    model.loss_fake = loss_fake
    #tf.summary.histogram("d_grads", d_grads)
    #tf.summary.histogram("g_grads", g_grads)
    model.saver = tf.train.Saver(max_to_keep=None)

    return d_optimize, g_optimize
