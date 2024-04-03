import os
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
from model.utils import Error, plotResults, get_train_name, print_log, PrintColor, preprocess_image
from model.config import Config
from model.loss import l2_loss, l1_loss, build_discriminator_loss, build_generator_loss, LPIPS_loss
from model.dataset_total import Dataset
import time
from model.utils import Conv, Upsample, Downsample, PRelu, FC
from tensorflow.python.keras.applications import vgg19

arg_scope = tf.contrib.framework.arg_scope


class STnet():
    def __int__(self):
        self.graph = tf.get_default_graph()
        self.saver = tf.train.Saver()

    def Gen(self, x, training_nn, scope):
        nlayers = [16, 64, 64, 96, ]

        x = tf.concat([x, tf.image.rgb_to_yuv(x)], axis=3)
        x0 = Conv(x, nlayers[1], scope + '/conv0', training_nn)
        # Block 1
        x1 = Conv(x0, nlayers[2], scope + '/conv1', training_nn)
        x1 = Conv(x1, nlayers[3], scope + '/conv2', training_nn)
        x1 = Downsample(x1, nlayers[2], scope + '/conv3', training_nn)
        # Block 2
        x2 = Conv(x1, nlayers[2], scope + '/conv4', training_nn)
        x2 = Conv(x2, nlayers[3], scope + '/conv5', training_nn)
        x2 = Downsample(x2, nlayers[2], scope + '/conv6', training_nn)
        # Block 3
        x3 = Conv(x2, nlayers[2], scope + '/conv7', training_nn)
        x3 = Conv(x3, nlayers[3], scope + '/conv8', training_nn)
        x3 = Downsample(x3, nlayers[2], scope + '/conv9', training_nn)
        # Decoder
        u1 = Upsample(x3, nlayers[1], scope + '/up1', training_nn)
        u2 = Upsample(tf.concat([u1, x2], 3), nlayers[1], scope + '/up2', training_nn)
        u3 = Upsample(tf.concat([u2, x1], 3), nlayers[1], scope + '/up3', training_nn)
        n1 = tf.nn.tanh(
            Conv(Conv(u1, nlayers[0], scope + '/n1', training_nn), 6, scope + '/nn1', training_nn, act=False,
                 norm=False))
        n2 = tf.nn.tanh(
            Conv(Conv(u2, nlayers[0], scope + '/n2', training_nn), 3, scope + '/nn2', training_nn, act=False,
                 norm=False))
        n3 = tf.nn.tanh(
            Conv(Conv(u3, nlayers[0], scope + '/n3', training_nn), 3, scope + '/nn3', training_nn, act=False,
                 norm=False))

        s = tf.reduce_mean(n1[:, :, :, 3:6], axis=[1, 2], keepdims=True)
        b = tf.reduce_mean(n1[:, :, :, :3], axis=[1, 2], keepdims=True)
        C = tf.nn.avg_pool(n2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        T = n3

        # ESR
        map1 = tf.image.resize_images(x1, [32, 32])
        map2 = tf.image.resize_images(x2, [32, 32])
        map3 = tf.image.resize_images(x3, [32, 32])
        maps = tf.concat([map1, map2, map3], 3)
        x4 = Conv(maps, nlayers[2], scope + '/conv10', training_nn, apply_dropout=True)
        x4 = Conv(x4, nlayers[1], scope + '/conv11', training_nn, apply_dropout=True)
        x5 = Conv(x4, 1, scope + '/conv12', training_nn, act=False, norm=False)

        return x5, s, b, C, T


class Synth():
    cnum = 32

    def __int__(self):
        self.graph = tf.get_default_graph()
        self.saver = tf.train.Saver()

    def _res_block(self, x, activation=tf.nn.leaky_relu, padding='SAME', name='res_block'):

        cnum = x.get_shape().as_list()[-1]
        xin = x
        x = tf.layers.conv2d(x, cnum // 4, kernel_size=1, strides=1, activation=activation, padding=padding,
                             name=name + '_conv1')
        x = tf.layers.conv2d(x, cnum // 4, kernel_size=3, strides=1, activation=activation, padding=padding,
                             name=name + '_conv2')
        x = tf.layers.conv2d(x, cnum, kernel_size=1, strides=1, activation=None, padding=padding, name=name + '_conv3')
        x = tf.add(xin, x, name=name + '_add')
        x = tf.layers.batch_normalization(x, name=name + '_bn')
        x = activation(x, name=name + '_out')
        return x

    def _conv_bn_relu(self, x, cnum=None, activation=tf.nn.leaky_relu, padding='SAME', name='conv_bn_relu'):

        cnum = x.get_shape().as_list()[-1] if cnum is None else cnum
        x = tf.layers.conv2d(x, cnum, kernel_size=3, strides=1, activation=None, padding=padding, name=name + '_conv')
        x = tf.layers.batch_normalization(x, name=name + '_bn')
        x = activation(x, name=name + '_out')
        return x

    def build_res_net(self, x, activation=tf.nn.leaky_relu, padding='SAME', name='res_net'):

        x = self._res_block(x, activation=activation, padding=padding, name=name + '_block1')
        x = self._res_block(x, activation=activation, padding=padding, name=name + '_block2')
        x = self._res_block(x, activation=activation, padding=padding, name=name + '_block3')
        x = self._res_block(x, activation=activation, padding=padding, name=name + '_block4')
        # x = self._res_block(x, activation=activation, padding=padding, name=name + '_block5')
        return x

    def build_encoder_net(self, x, activation=tf.nn.leaky_relu, padding='SAME', name='encoder_net',
                          get_feature_map=False):

        x = self._conv_bn_relu(x, self.cnum, name=name + '_conv1_1')
        x = self._conv_bn_relu(x, self.cnum, name=name + '_conv1_2')

        x = tf.layers.conv2d(x, 2 * self.cnum, kernel_size=3, strides=2, activation=activation, padding=padding,
                             name=name + '_pool1')
        x = self._conv_bn_relu(x, 2 * self.cnum, name=name + '_conv2_1')
        x = self._conv_bn_relu(x, 2 * self.cnum, name=name + '_conv2_2')
        f1 = x

        x = tf.layers.conv2d(x, 4 * self.cnum, kernel_size=3, strides=2, activation=activation, padding=padding,
                             name=name + '_pool2')
        x = self._conv_bn_relu(x, 4 * self.cnum, name=name + '_conv3_1')
        x = self._conv_bn_relu(x, 4 * self.cnum, name=name + '_conv3_2')
        f2 = x

        x = tf.layers.conv2d(x, 8 * self.cnum, kernel_size=3, strides=2, activation=activation, padding=padding,
                             name=name + '_pool3')
        x = self._conv_bn_relu(x, 8 * self.cnum, name=name + '_conv4_1')
        x = self._conv_bn_relu(x, 8 * self.cnum, name=name + '_conv4_2')
        if get_feature_map:
            return x, [f2, f1]
        else:
            return x

    def build_decoder_net(self, x, fuse=None, activation=tf.nn.leaky_relu, padding='SAME', name='decoder_net',
                          get_feature_map=False):

        if fuse and fuse[0] is not None:
            x = tf.concat([x, fuse[0]], axis=-1, name=name + '_fuse1')
        x = self._conv_bn_relu(x, 8 * self.cnum, name=name + '_conv1_1')
        x = self._conv_bn_relu(x, 8 * self.cnum, name=name + '_conv1_2')
        f1 = x

        x = tf.layers.conv2d_transpose(x, 4 * self.cnum, kernel_size=3, strides=2, activation=activation,
                                       padding=padding, name=name + '_deconv1')
        if fuse and fuse[1] is not None:
            x = tf.concat([x, fuse[1]], axis=-1, name=name + '_fuse2')
        x = self._conv_bn_relu(x, 4 * self.cnum, name=name + '_conv2_1')
        x = self._conv_bn_relu(x, 4 * self.cnum, name=name + '_conv2_2')
        f2 = x

        x = tf.layers.conv2d_transpose(x, 2 * self.cnum, kernel_size=3, strides=2, activation=activation,
                                       padding=padding, name=name + '_deconv2')
        if fuse and fuse[2] is not None:
            x = tf.concat([x, fuse[2]], axis=-1, name=name + '_fuse3')
        x = self._conv_bn_relu(x, 2 * self.cnum, name=name + '_conv3_1')
        x = self._conv_bn_relu(x, 2 * self.cnum, name=name + '_conv3_2')
        f3 = x

        x = tf.layers.conv2d_transpose(x, self.cnum, kernel_size=3, strides=2, activation=activation, padding=padding,
                                       name=name + '_deconv3')
        x = self._conv_bn_relu(x, self.cnum, name=name + '_conv4_1')
        x = self._conv_bn_relu(x, self.cnum, name=name + '_conv4_2')
        if get_feature_map:
            return x, [f1, f2, f3]
        else:
            return x

    def build_conversion_net(self, x_t, x_s, padding='SAME', name='tcn'):
        with tf.variable_scope('synthesis', reuse=tf.AUTO_REUSE):
            x_t = self.build_encoder_net(x_t, name=name + '_t_encoder')
            x_t = self.build_res_net(x_t, name=name + '_t_res')

            x_s = self.build_encoder_net(x_s, name=name + '_s_encoder')
            x_s = self.build_res_net(x_s, name=name + '_s_res')

            x = tf.concat([x_t, x_s], axis=-1, name=name + '_concat1')

            y_t = self.build_decoder_net(x, name=name + '_t_decoder')
            # y_t = tf.concat([y_sk, y_t], axis=-1, name=name + '_concat2')
            y_t = self._conv_bn_relu(y_t, name=name + '_t_cbr')
            y_t_out = tf.layers.conv2d(y_t, 3, kernel_size=3, strides=1, activation='tanh', padding=padding,
                                       name=name + '_t_out')
        return y_t_out


class CombineModel():
    def __init__(self):
        self.model1 = STnet()
        self.model2 = Synth()
        self.optimiezer = ''
        self.graph = tf.get_default_graph()
        self.cnum = 32
        with self.graph.as_default():
            self.BATCH_SIZE = Config.BATCH_SIZE
            self.IMAGE_SIZE = Config.IMAGE_SIZE
            self.IM2_SIZE = 160
            self.IM3_SIZE = 40
            self.global_step = tf.train.get_or_create_global_step()

    def generate(self, input, training_nn):
        bsize = self.BATCH_SIZE
        imsize = self.IMAGE_SIZE
        M, s, b, C, T = self.model1.Gen(input, training_nn, scope='STDN')

        trace1 = tf.image.resize_images(C, [imsize, imsize]) + T
        synth = self.model2.build_conversion_net(input[:bsize, ...], trace1[bsize:, ...])
        synth1 = input[:bsize, ...] + synth

        img_a = tf.stop_gradient(tf.concat([input[:bsize, ...], synth1], axis=0))
        M_s, s_s, b_s, C_s, T_s = self.model1.Gen(img_a, training_nn, scope='STDN')
        slive = (1 - s_s[bsize:, ...]) * synth1 - b_s[bsize:, ...] - \
                tf.image.resize_images(C_s[bsize:, ...],[imsize, imsize]) - T_s[bsize:,...]

        return s, b, C, T, synth, synth1, slive, C_s, T_s

    def Disc_s(self, x, training_nn, scope):
        nlayers = [16, 32, 64, 96, ]
        x = tf.concat([x, tf.image.rgb_to_yuv(x)], axis=3)
        # Block 1
        # x1 = Conv(x, nlayers[1], scope+'/conv1', training_nn)
        x1 = Downsample(x, nlayers[1], scope + '/conv2', training_nn)
        # Block 2
        # x2 = Conv(x1, nlayers[2], scope+'/conv3', training_nn)
        x2 = Downsample(x1, nlayers[2], scope + '/conv4', training_nn)
        # Block 3
        # x3 = Conv(x2, nlayers[2], scope+'/conv5', training_nn)
        x3 = Downsample(x2, nlayers[3], scope + '/conv6', training_nn)
        # Block 4
        x4 = Conv(x3, nlayers[3], scope + '/conv7', training_nn)
        x4l = Conv(x4, 1, scope + '/conv8', training_nn, act=False, norm=False)
        x4s = Conv(x4, 1, scope + '/conv9', training_nn, act=False, norm=False)
        return x4l, x4s

    def build_discriminator(self, x, activation=tf.nn.leaky_relu, padding='SAME', name='discriminator'):
        with tf.variable_scope('D1', reuse=tf.AUTO_REUSE):
            x = tf.layers.conv2d(x, 64, kernel_size=3, strides=2, activation=activation, padding=padding,
                                 name=name + '_conv1')
            x = tf.layers.conv2d(x, 128, kernel_size=3, strides=2, activation=None, padding=padding,
                                 name=name + '_conv2')
            x = tf.layers.batch_normalization(x, name=name + '_conv2_bn')
            x = activation(x, name=name + '_conv2_activation')
            x = tf.layers.conv2d(x, 256, kernel_size=3, strides=2, activation=None, padding=padding,
                                 name=name + '_conv3')
            x = tf.layers.batch_normalization(x, name=name + '_conv3_bn')
            x = activation(x, name=name + '_conv3_activation')
            x = tf.layers.conv2d(x, 512, kernel_size=3, strides=2, activation=None, padding=padding,
                                 name=name + '_conv4')
            x = tf.layers.batch_normalization(x, name=name + '_conv4_bn')
            x = activation(x, name=name + '_conv4_activation')
            x = tf.layers.conv2d(x, 1, kernel_size=3, strides=1, activation=None, padding=padding, name=name + '_conv5')
            x = tf.layers.batch_normalization(x, name=name + '_conv5_bn')
            x = tf.nn.sigmoid(x, name='_out')
        return x

    def get_train_op(self, sum_loss, global_step, config, scope_name):

        decay_steps = config.NUM_EPOCHS_PER_DECAY * config.STEPS_PER_EPOCH

        self.lr = tf.train.exponential_decay(config.LEARNING_RATE,
                                             global_step,
                                             decay_steps,
                                             config.LEARNING_RATE_DECAY_FACTOR,
                                             staircase=True)

        # Generate moving averages of all losses and associated summaries.
        # 对损失函数进行滑动平均处理，减少波动，提高模型泛化能力
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        loss_averages_op = loss_averages.apply([sum_loss])
        # Compute gradients.
        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.AdamOptimizer(self.lr)
            grads = opt.compute_gradients(sum_loss, var_list=tf.trainable_variables(scope=scope_name))
        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Track the moving averages of all trainable variables.
        with tf.name_scope('TRAIN-' + scope_name) as scope:
            variable_averages = tf.train.ExponentialMovingAverage(config.MOVING_AVERAGE_DECAY, global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables(scope=scope_name))

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train-' + scope_name)

        return train_op

    def step(self, config, data_batch, training_nn):
        global_step = self.global_step

        # Get images and labels.
        img = data_batch.nextit

        imsize = self.IMAGE_SIZE
        im2size = self.IM2_SIZE
        im3size = self.IM3_SIZE
        bsize = self.BATCH_SIZE

        img = tf.transpose(img, perm=[1, 0, 2, 3, 4])
        img = tf.reshape(img, [bsize * 2, imsize, imsize, 3])
        img2 = tf.image.resize_images(img, [im2size, im2size])
        img3 = tf.image.resize_images(img, [im3size, im3size])

        s, b, C, T, trace_wrap, synth1, slive, C_s, T_s = self.generate(img, training_nn)  #
        recon1 = (1 - s) * img - b - tf.image.resize_images(C, [imsize, imsize]) - T
        trace = img - recon1
        trace1 = tf.image.resize_images(C, [imsize, imsize]) + T

        img_d1 = tf.concat([img, recon1[bsize:, ...], synth1], 0)
        d1l, d1s = self.Disc_s(img_d1, training_nn=training_nn, scope='Disc/d1')

        recon2 = tf.image.resize_images(recon1, [im2size, im2size])
        synth2 = tf.image.resize_images(synth1, [im2size, im2size])
        img_d2 = tf.concat([img2, recon2[bsize:, ...], synth2], 0)
        d2l, d2s = self.Disc_s(img_d2, training_nn=training_nn, scope='Disc/d2')

        recon3 = tf.image.resize_images(recon1, [im3size, im3size])
        synth3 = tf.image.resize_images(synth1, [im3size, im3size])
        img_d3 = tf.concat([img3, recon3[bsize:, ...], synth3], 0)
        d3l, d3s = self.Disc_s(img_d3, training_nn=training_nn, scope='Disc/d3')

        C_N = tf.image.resize_images(C, [imsize, imsize])
        T2 = tf.image.resize_images(T, [im2size, im2size])
        T3 = tf.image.resize_images(T, [im3size, im3size])
        C_N2 = tf.image.resize_images(C_N, [im2size, im2size])
        C_N3 = tf.image.resize_images(C_N, [im3size, im3size])

        d1_rl, _, d1_sl, _ = tf.split(d1l, 4)
        d2_rl, _, d2_sl, _ = tf.split(d2l, 4)
        d3_rl, _, d3_sl, _ = tf.split(d3l, 4)
        _, d1_rs, _, d1_ss = tf.split(d1s, 4)
        _, d2_rs, _, d2_ss = tf.split(d2s, 4)
        _, d3_rs, _, d3_ss = tf.split(d3s, 4)

        gan_loss1 = l2_loss(d1_sl, 1) + l2_loss(d2_sl, 1) + l2_loss(d3_sl, 1)

        gan_loss2 = l2_loss(d1_ss, 1) + l2_loss(d2_ss, 1) + l2_loss(d3_ss, 1)


        reg_loss_li = l2_loss(s[:bsize, ...], 0) + l2_loss(b[:bsize, ...], 0) + \
                      l2_loss(C[:bsize, ...], 0) + l2_loss(T[:bsize, ...], 0)
        reg_loss_sp = l2_loss(s[bsize:, ...], 0) + l2_loss(b[bsize:, ...], 0) + \
                      l2_loss(C[bsize:, ...], 0) + l2_loss(T[bsize:, ...], 0)
        reg_loss = reg_loss_li * 10 + reg_loss_sp * 1e-4  # 10 -4


        g_loss = gan_loss1 + reg_loss  # +gan_loss2# + lt_l1 + lc_l1


        d_loss = (l2_loss(d1_rl, 1) + l2_loss(d2_rl, 1) + l2_loss(d3_rl, 1) + \
                  l2_loss(d1_rs, 1) + l2_loss(d2_rs, 1) + l2_loss(d3_rs, 1) + \
                  l2_loss(d1_sl, 0) + l2_loss(d2_sl, 0) + l2_loss(d3_sl, 0) + \
                  l2_loss(d1_ss, 0) + l2_loss(d2_ss, 0) + l2_loss(d3_ss, 0)) / 4


        pixel_loss = l1_loss(slive, img[:bsize, ...])

        C_N_s = tf.image.resize_images(C_s, [imsize, imsize])

        T2_s = tf.image.resize_images(T_s, [im2size, im2size])
        T3_s = tf.image.resize_images(T_s, [im3size, im3size])
        C_N2_s = tf.image.resize_images(C_N_s, [im2size, im2size])
        C_N3_s = tf.image.resize_images(C_N_s, [im3size, im3size])


        trace_wrap2 = tf.image.resize_images(trace_wrap, [im2size, im2size])
        trace_wrap3 = tf.image.resize_images(trace_wrap, [im3size, im3size])

        img_d7 = tf.concat([T[bsize:, ...] + C_N[bsize:, ...], trace_wrap], 0)
        d7l, d7s = self.Disc_s(img_d7, training_nn=training_nn, scope='Disc/d1')

        img_d8 = tf.concat([T2[bsize:, ...] + C_N2[bsize:, ...], trace_wrap2], 0)
        d8l, d8s = self.Disc_s(img_d8, training_nn=training_nn, scope='Disc/d2')

        img_d9 = tf.concat([T3[bsize:, ...] + C_N3[bsize:, ...], trace_wrap3], 0)
        d9l, d9s = self.Disc_s(img_d9, training_nn=training_nn, scope='Disc/d3')

        d1_rt2, d1_st2 = tf.split(d7l, 2)
        d2_rt2, d2_st2 = tf.split(d8l, 2)
        d3_rt2, d3_st2 = tf.split(d9l, 2)


        if training_nn:
            g_op = self.get_train_op(g_loss, global_step, config, "STDN")
            d_op = self.get_train_op(d_loss, global_step, config, "Disc")
            g2_op = self.get_train_op(gan_loss2 + 10 * pixel_loss, global_step, config, "synthesis")

        else:
            g_op = None
            d_op = None
            g2_op = None
            # d2_op = None

        # log info
        losses = [g_loss, d_loss, gan_loss1, reg_loss]  # , pixel_loss, gan_loss2
        fig = [img, s, b, C * 10, T * 10, recon1, tf.concat([trace1[bsize:, ...] * 10, trace_wrap * 10], 0),
               tf.concat([slive, synth1], 0),
               tf.concat([(T_s[bsize:, ...]) * 10, (tf.image.resize_images(C_s[bsize:, ...], [imsize, imsize])) * 10],
                         0)]
        fig = plotResults(fig)
        return losses, g_op, d_op, g2_op, fig


def main(argv=None):
    # Configurations
    config = Config(gpu='1',
                    root_dir='/data/data5/multimodal/D4/train/',
                    root_dir_val='/data/data5/multimodal/D4/val/',
                    mode='training')

    if not config.train_name:
        train_name = get_train_name()
    else:
        train_name = config.train_name

    # Create data feeding pipeline.
    model = CombineModel()
    dataset_train = Dataset(config, 'train', train_name)
    dataset_val = Dataset(config, 'val', train_name)

    # Train Graph
    losses, g_op, d_op, g2_op,fig = model.step(config, dataset_train, training_nn=True)  # g2_op,
    losses_val, _, _, _, fig_val = model.step(config, dataset_val, training_nn=False)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(max_to_keep=50, )
    with tf.Session(config=config.GPU_CONFIG) as sess:
        # Restore the model
        ckpt = tf.train.get_checkpoint_state(config.checkpoint)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            last_epoch = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print('**********************************************************')
            print('Restore from Epoch ' + str(last_epoch))
            print('**********************************************************')
        else:
            init = tf.initializers.global_variables()
            last_epoch = 0
            sess.run(init)
            print('**********************************************************')
            print('Train from scratch.')
            print('**********************************************************')

            avg_loss = Error()
            print_list = {}
            for epoch in range(int(last_epoch), config.MAX_EPOCH):
                start = time.time()
                # Train one epoch
                for step in range(config.STEPS_PER_EPOCH):
                    if step % config.G_D_RATIO == 0:
                        _losses = sess.run(losses + [g_op,g2_op, d_op, fig])  #
                    else:
                        _losses = sess.run(losses + [g_op,g2_op, fig])  # g2_op,

                    # Logging
                    print_list['g_loss'] = _losses[0]
                    print_list['d_loss'] = _losses[1]
                    print_list['gan_loss1'] = _losses[2]
                    print_list['reg_loss'] = _losses[3]
                    print_list['pixel_loss'] = _losses[4]
                    print_list['gan_loss2'] = _losses[5]
                    display_list = ['Epoch ' + str(epoch + 1) + '-' + str(step + 1) + '/' + str(
                        config.STEPS_PER_EPOCH) + ':'] + \
                                   [avg_loss(x) for x in print_list.items()]
                    print(*display_list + ['          '], end='\r')
                    # Visualization
                    if step % config.LOG_FR_TRAIN == 0:
                        if not os.path.exists(config.LOG_DIR + '/' + train_name):
                            os.makedirs(config.LOG_DIR + '/' + train_name)
                        savedir = os.path.join(config.LOG_DIR, train_name)
                        fname = savedir + '/Epoch-' + str(epoch + 1) + '-' + str(step + 1) + '.png'
                        cv2.imwrite(fname, _losses[-1])

                # Model saving
                if not os.path.exists(config.checkpoint + '/' + train_name):
                    os.makedirs(config.checkpoint + '/' + train_name)
                savedir_ckpt = os.path.join(config.checkpoint, train_name)
                saver.save(sess, savedir_ckpt + '/ckpt', global_step=epoch + 1)
                print('\n', end='\r')

                # Validate one epoch
                for step in range(config.STEPS_PER_EPOCH_VAL):

                    _losses = sess.run(losses_val + [fig_val])

                    # Logging
                    print_list['g_loss'] = _losses[0]
                    print_list['d_loss'] = _losses[1]
                    print_list['gan_loss1'] = _losses[2]
                    print_list['reg_loss'] = _losses[3]
                    print_list['pixel_loss'] = _losses[4]
                    print_list['gan_loss2'] = _losses[5]
                    display_list = ['Epoch ' + str(epoch + 1) + '-Val-' + str(step + 1) + '/' + str(
                        config.STEPS_PER_EPOCH_VAL) + ':'] + \
                                   [avg_loss(x, val=1) for x in print_list.items()]
                    print(*display_list + ['          '], end='\r')
                    # Visualization
                    if step % config.LOG_FR_TEST == 0:
                        fname = savedir + '/Epoch-' + str(epoch + 1) + '-Val-' + str(step + 1) + '.png'
                        cv2.imwrite(fname, _losses[-1])

                # time of one epoch
                print('\n    Time taken for epoch {} is {:3g} sec'.format(epoch + 1, time.time() - start))
                avg_loss.reset()


if __name__ == '__main__':
    tf.app.run()
