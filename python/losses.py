# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

e = 1e-8

def kld_unit_mvn(mu, log_var):
        kl_loss = - 0.5 * tf.reduce_mean(log_var - tf.square(mu) - tf.exp(log_var) + 1)
        return kl_loss

def loss_q_logp(mu1,var1,mu2,var2):
    return -0.5*(tf.reduce_mean(1+tf.math.log(var1+e))) + 0.5*(tf.reduce_mean(tf.math.log(var2+e)) + tf.reduce_mean((var1+e)/(var2+e)) + tf.reduce_mean((1.0 / (var2+e)) * (mu1 - mu2)*(mu1 - mu2)))

def log_diag_mvn(mu, var):
    def f(x):
        # expects batches
        k = tf.shape(mu)[1]

        logp = (-k / 2.0) * tf.math.log(2 * np.pi) - 0.5 * tf.reduce_mean(tf.math.log(var)) - tf.reduce_mean(0.5 * (1.0 / var) * tf.square(x - mu))
        return -logp
    return f

def binary_crossentropy(inputs,x_hat):
    cross_ent = - tf.reduce_sum(inputs * tf.math.log(1e-6 + x_hat) \
                   + (1 - inputs) * tf.math.log(1e-6 + 1 - x_hat))
    return cross_ent

def compute_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

def compute_mmd(x, y, sigma_sqr=1.0):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)

    return -tf.reduce_sum(-0.5 * ((sample - mean) ** 2. * tf.math.exp(-logvar) + logvar + log2pi),axis=raxis)
