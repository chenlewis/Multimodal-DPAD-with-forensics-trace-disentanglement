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
from __future__ import absolute_import, division, print_function, unicode_literals

import os.path

import tensorflow as tf
import time
import numpy as np
from matplotlib import pyplot as plt
import cv2

import write_in_csv
from model.dataset_TC import Dataset
from model.config import Config
from real_data_model import CombineModel
from model.utils import get_train_name
from skimage import io, util
from model.utils import Error, plotResults
from tqdm import tqdm


def _step(config, data_batch,model,training_nn):
    global_step = tf.train.get_or_create_global_step()
    bsize = config.BATCH_SIZE
    imsize = config.IMAGE_SIZE

    # Get images and labels for CNN.
    img,im_name_li, im_name_sp = data_batch.nextit
    img = tf.reshape(img, [bsize*2, imsize, imsize, 3])

    # Forward the Generator
    s, b, C, T, trace_wrap, synth1, slive, C_s, T_s = model.generate(input=img, training_nn=False)
    recon1 = (1 - s) * img - b - tf.image.resize_images(C, [imsize, imsize]) - T
    synthesis_C = img[:bsize, ...]+tf.image.resize_images(C_s[bsize:, ...], [imsize, imsize])
    synthesis_T = img[:bsize, ...]+T_s[bsize:, ...]

    no_synthesis_C = img[:bsize, ...] + tf.image.resize_images(C[bsize:, ...], [imsize, imsize])
    no_synthesis_T = img[:bsize, ...] + T[bsize:, ...]
    no_synthesis_G = img[:bsize, ...] + tf.image.resize_images(C[bsize:, ...], [imsize, imsize]) + T[bsize:, ...]

    G = tf.image.resize_images(C, [imsize, imsize]) + T
    G_live = G[:bsize, ...]
    G_spoof = G[bsize:, ...]
    # 前段：img[:bsize, ...]；后段：img[bsize:, ...]
    # live
    C_l = tf.image.resize_images(C[:bsize, ...], [imsize, imsize])
    T_l = T[:bsize, ...]
    # spoof
    C_s = tf.image.resize_images(C[bsize:, ...], [imsize, imsize])
    T_s = T[bsize:, ...]

    return G_spoof,recon1[bsize:, ...],recon1[:bsize, ...],synth1,synthesis_C,synthesis_T,im_name_li,im_name_sp


def main(argv=None):
    # Configurations
    config = Config(gpu='1',
                    root_dir='/data/data5/yi/train/',
                    root_dir_val=None,
                    mode='testing')
    config.BATCH_SIZE = 1

    # Get images and labels.
    if not config.train_name:
        train_name = get_train_name()
    else:
        train_name = config.train_name

    dataset_test = Dataset(config,'test',train_name)

    # 加载模型
    model = CombineModel()
    _G_spoof,_recon1,_recon1_live,_synth1,_synthesis_C,_synthesis_T,_im_name_li,_im_name_sp = _step(config, dataset_test, model,training_nn=False)
    print('begin')

    image_li_list = []
    image_sp_list = []
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(max_to_keep=50, )
    with tf.Session(config=config.GPU_CONFIG) as sess:
        # Restore the model
        ckpt = tf.train.get_checkpoint_state(config.checkpoint)
        print(ckpt)
        print(ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            last_epoch = ckpt.model_checkpoint_path  # .split('/')[-1].split('-')[-1]
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

        step_per_epoch = int(len(dataset_test.name_list_li) / config.BATCH_SIZE)
        for step in tqdm(range(step_per_epoch), desc="Generating Images"):
            G_spoof,recon1,recon1_live,synth1,synthesis_C,synthesis_T,im_name_li,im_name_sp \
                = sess.run([_G_spoof,_recon1,_recon1_live,_synth1,_synthesis_C,_synthesis_T,_im_name_li,_im_name_sp])
            im_name_li_str = im_name_li[0].decode('utf-8')
            img_name_li = os.path.basename(im_name_li_str.split('\\')[-1])

            im_name_sp_str = im_name_sp[0].decode('utf-8')
            img_name_sp = os.path.basename(im_name_sp_str.split('\\')[-1])


            recon1 = recon1 * 255 / np.max(recon1)
            synth1 = synth1 * 255 / np.max(synth1)
            synthesis_C = synthesis_C * 255 / np.max(synthesis_C)
            synthesis_T = synthesis_T * 255 / np.max(synthesis_T)
            recon1_live = recon1_live * 255 / np.max(recon1_live)

            folder_path5 = '/data/data6/yi/data5_ECCV/result/20230713103633_1018_train/live_to_spoof/'
            folder_path6 = '/data/data6/yi/data5_ECCV/result/20230713103633_1018_train/spoof_to_live/'
            folder_path7 = '/data/data6/yi/data5_ECCV/result/20230713103633_1018_train/synthesis_C/'
            folder_path8 = '/data/data6/yi/data5_ECCV/result/20230713103633_1018_train/synthesis_T/'
            os.makedirs(folder_path5, exist_ok=True)
            os.makedirs(folder_path6, exist_ok=True)
            os.makedirs(folder_path7, exist_ok=True)
            os.makedirs(folder_path8, exist_ok=True)
            folder_path9 = '/data/data6/yi/data5_ECCV/result/20230713103633_1018_train/live_to_live/'
            os.makedirs(folder_path9, exist_ok=True)
            # 合成的翻拍图像
            synth_rgb = cv2.cvtColor(synth1[0], cv2.COLOR_BGR2RGB)
            cv2.imwrite(folder_path5 + img_name_sp.split('.')[0] + '.png', synth_rgb)

            # 合成 only C
            synthesis_C_rgb = cv2.cvtColor(synthesis_C[0], cv2.COLOR_BGR2RGB)
            cv2.imwrite(folder_path7 + img_name_sp.split('.')[0] + '.png', synthesis_C_rgb)

            # 合成 only T
            synthesis_T_rgb = cv2.cvtColor(synthesis_T[0], cv2.COLOR_BGR2RGB)
            cv2.imwrite(folder_path8 + img_name_sp.split('.')[0] + '.png', synthesis_T_rgb)

            # 真实翻拍图像解离后的伪真实图像
            recon1_rgb = cv2.cvtColor(recon1[0], cv2.COLOR_BGR2RGB)
            cv2.imwrite(folder_path6 + img_name_sp.split('.')[0] + '.png', recon1_rgb)

            recon1_live_rgb = cv2.cvtColor(recon1_live[0], cv2.COLOR_BGR2RGB)
            cv2.imwrite(folder_path9 + img_name_li.split('.')[0] + '.png', recon1_live_rgb)

            image_li_list.append(img_name_li)
            image_sp_list.append(img_name_sp)


            G_live = G_live * 255 / np.max(G_live)
            G_spoof = G_spoof * 255 / np.max(G_spoof)

            folder_path10 = '/data/data6/yi/data5_ECCV/result/20230713103633_1018_train/G_live/'
            os.makedirs(folder_path10, exist_ok=True)
            folder_path11 = '/data/data6/yi/data5_ECCV/result/20230713103633_1018_train/G_spoof/'
            os.makedirs(folder_path11, exist_ok=True)

            G_live_rgb = cv2.cvtColor(G_live[0], cv2.COLOR_BGR2RGB)
            cv2.imwrite(folder_path10 + img_name_li.split('.')[0] + '.png', G_live_rgb)
            G_spoof_rgb = cv2.cvtColor(G_spoof[0], cv2.COLOR_BGR2RGB)
            cv2.imwrite(folder_path11 + img_name_sp.split('.')[0] + '.png', G_spoof_rgb)


        print("\n")
    print('over')


if __name__ == '__main__':
    tf.app.run()
