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
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from datetime import datetime

arg_scope = tf.contrib.framework.arg_scope

PrintColor = {
    'black': 30,
    'red': 31,
    'green': 32,
    'yellow': 33,
    'blue': 34,
    'amaranth': 35,
    'ultramarine': 36,
    'white': 37
}

PrintStyle = {
    'default': 0,
    'highlight': 1,
    'underline': 4,
    'flicker': 5,
    'inverse': 7,
    'invisible': 8
}


def plotResults(result_list):
    column = []
    for fig in result_list:
        shape = fig.shape
        fig = tf.clip_by_value(fig, 0.0, 1.0)  # 把fig中每个元素的值都压缩在0和1之间
        row = []
        if fig.shape[3] == 1:
            fig = tf.concat([fig, fig, fig], axis=3)
        else:
            r, g, b = tf.split(fig, 3, 3)  # 对维度3切割3份
            fig = tf.concat([b, g, r], 3)
        fig = tf.image.resize_images(fig, [224, 224])
        row = tf.split(fig, shape[0])
        row = tf.concat(row, axis=2)
        column.append(row[0, :, :, :])

    column = tf.concat(column, axis=0)
    img = tf.cast(column * 255, tf.uint8)
    return img


def plotResults_test(result_list):
    fig = result_list[0]
    fig = tf.clip_by_value(fig, 0.0, 1.0)
    if fig.shape[3] == 1:
        fig = tf.concat([fig, fig, fig], axis=3)
    else:
        r, g, b = tf.split(fig, 3, 3)
        fig = tf.concat([b, g, r], 3)
    fig = tf.image.resize_images(fig, [224, 224])
    img = tf.cast(fig * 255, tf.uint8)
    return img


class Error:
    def __init__(self):
        self.losses = {}

    def __call__(self, update, val=0):
        loss_name = update[0]
        loss_update = update[1]
        if loss_name not in self.losses.keys():
            self.losses[loss_name] = {'value': 0, 'step': 0, 'value_val': 0, 'step_val': 0}
        if val == 1:
            if loss_update is not None:
                self.losses[loss_name]['value_val'] += loss_update
                self.losses[loss_name]['step_val'] += 1
            smooth_loss = str(
                np.around(self.losses[loss_name]['value_val'] / (self.losses[loss_name]['step_val'] + 1e-5),
                          decimals=3))
            return loss_name + ':' + smooth_loss + ','
        else:
            if loss_update is not None:
                self.losses[loss_name]['value'] = self.losses[loss_name]['value'] * 0.9 + loss_update * 0.1
                self.losses[loss_name]['step'] += 1
            if self.losses[loss_name]['step'] == 1:
                self.losses[loss_name]['value'] = loss_update
            smooth_loss = str(np.around(self.losses[loss_name]['value'], decimals=3))
            return loss_name + ':' + smooth_loss + ','

    def reset(self):
        self.losses = {}


def PRelu(x, scope):
    with tf.variable_scope(scope + '/PRelu', reuse=tf.AUTO_REUSE):
        alphas = tf.get_variable('alpha', x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        x = tf.nn.relu(x) + alphas * (x - abs(x)) * 0.5
    return x


def FC(x, num, scope, training_nn, act=True, norm=True, apply_dropout=False):
    with arg_scope([layers.fully_connected],
                   weights_initializer=tf.random_normal_initializer(stddev=0.02),
                   biases_initializer=tf.constant_initializer(0.0),
                   activation_fn=None,
                   normalizer_fn=None,
                   trainable=training_nn,
                   reuse=tf.AUTO_REUSE):
        x = layers.fully_connected(x, num_outputs=num, scope=scope)
        if norm:
            x = layers.batch_norm(x, decay=0.99,
                                  scale=True,
                                  epsilon=1e-5,
                                  is_training=training_nn,
                                  updates_collections=None,
                                  reuse=tf.AUTO_REUSE,
                                  scope=scope + '/BN')
        if act:
            x = PRelu(x, scope)
        if apply_dropout:
            x = layers.dropout(x, keep_prob=0.7, is_training=training_nn, scope=scope + '/dropout')
    return x


def Conv(x, num, scope, training_nn, act=True, norm=True, apply_dropout=False, padding='SAME', stride=1):
    with arg_scope([layers.conv2d],
                   kernel_size=3,
                   weights_initializer=tf.random_normal_initializer(stddev=0.02),
                   biases_initializer=tf.constant_initializer(0.0),
                   activation_fn=None,
                   normalizer_fn=None,
                   trainable=training_nn,
                   padding=padding,
                   reuse=tf.AUTO_REUSE,
                   stride=stride):
        x = layers.conv2d(x, num_outputs=num, scope=scope)
        if norm:
            x = layers.batch_norm(x, decay=0.99,
                                  scale=True,
                                  epsilon=1e-5,
                                  is_training=training_nn,
                                  updates_collections=None,
                                  reuse=tf.AUTO_REUSE,
                                  scope=scope + '/BN')
        if act:
            x = PRelu(x, scope)
        if apply_dropout:
            x = layers.dropout(x, keep_prob=0.7, is_training=training_nn, scope=scope + '/dropout')
    return x


def Downsample(x, num, scope, training_nn, padding='SAME', act=True, norm=True, apply_dropout=False):
    with arg_scope([layers.conv2d],
                   kernel_size=3,
                   weights_initializer=tf.random_normal_initializer(stddev=0.02),
                   biases_initializer=tf.constant_initializer(0.0),
                   activation_fn=None,
                   normalizer_fn=None,
                   trainable=training_nn,
                   padding=padding,
                   reuse=tf.AUTO_REUSE,
                   stride=2):
        x = layers.conv2d(x, num_outputs=num, scope=scope)
        if norm:
            x = layers.batch_norm(x, decay=0.99,
                                  scale=True,
                                  epsilon=1e-5,
                                  is_training=training_nn,
                                  updates_collections=None,
                                  reuse=tf.AUTO_REUSE,
                                  scope=scope + '/BN')
        if act:
            x = PRelu(x, scope)
        if apply_dropout:
            x = layers.dropout(x, keep_prob=0.7, is_training=training_nn, scope=scope + '/dropout')
    return x


def Upsample(x, num, scope, training_nn, padding='SAME', act=True, norm=True):
    with arg_scope([layers.conv2d_transpose],
                   kernel_size=3,
                   weights_initializer=tf.random_normal_initializer(stddev=0.02),
                   biases_initializer=tf.constant_initializer(0.0),
                   activation_fn=None,
                   normalizer_fn=None,
                   trainable=training_nn,
                   padding=padding,
                   reuse=tf.AUTO_REUSE,
                   stride=2):
        x = layers.conv2d_transpose(x, num_outputs=num, scope=scope)
        if norm:
            x = layers.batch_norm(x, decay=0.99,
                                  scale=True,
                                  epsilon=1e-5,
                                  is_training=training_nn,
                                  updates_collections=None,
                                  reuse=tf.AUTO_REUSE,
                                  scope=scope + '/BN')
        if act:
            x = PRelu(x, scope)
    return x


def get_train_name():
    # get current time for train name
    return datetime.now().strftime('%Y%m%d%H%M%S')


def print_log(s, time_style=PrintStyle['default'], time_color=PrintColor['blue'],
              content_style=PrintStyle['default'], content_color=PrintColor['white']):
    # colorful print s with time log
    cur_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    log = '\033[{};{}m[{}]\033[0m \033[{};{}m{}\033[0m'.format \
        (time_style, time_color, cur_time, content_style, content_color, s)
    print(log)


def preprocess_image(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_images(image, (224, 224))
    return image
