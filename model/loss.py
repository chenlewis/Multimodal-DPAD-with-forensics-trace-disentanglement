# Copyright 2020
# 
# Yaojie Liu, Joel Stehouwer, Xiaoming Liu, Michigan State University
# 
# All Rights Reserved.
# 
# This research is based upon work supported by the Office of the Director of 
# National Intelligence (ODNI), Intelligence Advanced Research Projects Activity
# (IARPA), via IARPA R&D Contract No. 2017-17020200004. The views and 
# conclusions contained herein are those of the authors and should not be 
# interpreted as necessarily representing the official policies or endorsements,
# either expressed or implied, of the ODNI, IARPA, or the U.S. Government. The 
# U.S. Government is authorized to reproduce and distribute reprints for 
# Governmental purposes not withstanding any copyright annotation thereon. 
# ==============================================================================
import os

import tensorflow as tf
from model.config import Config


def l1_loss(x, y, mask=None):
    xshape = x.shape
    if mask is not None:
        loss = tf.math.reduce_mean(tf.reshape(tf.abs(x - y), [xshape[0], -1]), axis=1, keepdims=True)
        loss = tf.math.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-8)
    else:
        loss = tf.math.reduce_mean(tf.abs(x - y))
    return loss


def l2_loss(x, y, mask=None):
    xshape = x.shape
    if mask is not None:
        loss = tf.math.reduce_mean(tf.reshape(tf.square(x - y), [xshape[0], -1]), axis=1, keepdims=True)
        loss = tf.math.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-8)
    else:
        loss = tf.math.reduce_mean(tf.square(x - y))
    return loss


def build_l1_loss(x_t, x_o, name='l1_loss'):
    return tf.reduce_mean(tf.abs(x_t - x_o))


def build_discriminator_loss(x, name='d_loss'):
    x_true, x_pred = tf.split(x, 2, name=name + '_split')
    d_loss = -tf.reduce_mean(tf.math.log(tf.clip_by_value(x_true, Config.epsilon, 1.0)) \
                             + tf.math.log(tf.clip_by_value(1.0 - x_pred, Config.epsilon, 1.0)))
    return d_loss


def build_gan_loss(x, name='gan_loss'):
    x_true, x_pred = tf.split(x, 2, name=name + '_split')
    gan_loss = -tf.reduce_mean(tf.math.log(tf.clip_by_value(x_pred, Config.epsilon, 1.0)))
    return gan_loss


def build_generator_loss(out_d, out_g, labels, name='g_loss'):
    o_db = out_d
    o_b = out_g
    # o_vgg = out_vgg
    t_b = labels

    l_b_gan = build_gan_loss(o_db, name=name + '_ls_gan_loss')
    l_b_l1 = Config.ls_beta * l2_loss(o_b, t_b)  # , name=name + '_ls_l1_loss'
    l_b = l_b_gan + l_b_l1

    return l_b, l_b_gan, l_b_l1


def build_generator_loss_T(out_d, out_g, labels, name='g_loss'):
    o_tb = out_d  # o_t
    o_t = out_g  # T[bsize:, ...]
    t_t = labels  # img_sv[:bsize, ...]

    l_t_gan = build_gan_loss(o_tb, name=name + '_ls_gan_loss')
    l_t_l1 = Config.lt_beta * build_l1_loss(t_t, o_t, name=name + '_ls_l1_loss')
    # l_t_l1 = Config.lt_beta * l2_loss(o_t, t_t)
    l_t = l_t_gan + l_t_l1

    return l_t, l_t_gan, l_t_l1


def build_generator_loss_C(out_d, out_g, labels, name='g_loss'):
    o_cb = out_d  # o_c
    o_c = out_g  # C_N[bsize:, ...]
    t_c = labels  # img_sv[bsize:, ...]

    l_c_gan = build_gan_loss(o_cb, name=name + '_ls_gan_loss')
    l_c_l1 = Config.lc_beta * build_l1_loss(t_c, o_c, name=name + '_ls_l1_loss')
    # l_c_l1 = Config.lc_beta * l2_loss(o_c, t_c)
    l_c = l_c_gan + l_c_l1

    return l_c, l_c_gan, l_c_l1


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(disc_generated_output), disc_generated_output)

    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (200 * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


def calculate_distance(T_feature, C_feature, weight):
    return tf.norm(tf.subtract(T_feature, C_feature), ord='euclidean') * weight


def LPIPS_loss(T_features, C_features, perceptual_weights):
    perceptual_distances = tf.map_fn(lambda x: calculate_distance(*x), (T_features, C_features, perceptual_weights),
                                     dtype=tf.float32)

    lpips_distance = tf.reduce_sum(perceptual_distances)

    return lpips_distance

