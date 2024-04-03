import os

import cv2
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
import tensorflow as tf
import numpy as np
import random
import glob
from PIL import Image
from write_in_csv import write_in_csv2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
autotune = -1


class Dataset():
    def __init__(self, config, train_mode,train_name):
        self.config = config
        self.train_name = train_name
        if self.config.MODE == 'training':
            self.input_tensors = self.inputs_for_training(train_mode)
        elif self.config.MODE == 'testing':
            self.input_tensors, self.name_list = self.inputs_for_testing()
        self.nextit = self.input_tensors.make_one_shot_iterator().get_next()
        # 调用后能读取数据

    def inputs_for_training(self, train_mode):

        global data_dir_li, data_dir_sp, data_dir_st, data_dir_sc
        if train_mode == 'train':
            data_dir_li = self.config.LI_DATA_DIR
            data_dir_sp = self.config.SP_DATA_DIR
            #data_dir_rs = self.config.RS_DATA_DIR
        elif train_mode == 'val':
            data_dir_li = self.config.LI_DATA_DIR_VAL
            data_dir_sp = self.config.SP_DATA_DIR_VAL
            #data_dir_rs = self.config.RS_DATA_DIR_VAL
        li_data_samples = []
        for _dir in data_dir_li:
            _list = glob.glob(_dir)
            li_data_samples += _list
        sp_data_samples = []
        for _dir in data_dir_sp:
            _list = glob.glob(_dir)
            sp_data_samples += _list

        # make live/spoof sample lists equal

        li_len = len(li_data_samples)
        sp_len = len(sp_data_samples)
        if li_len < sp_len:
            while len(li_data_samples) < sp_len:
                li_data_samples += random.sample(li_data_samples, len(li_data_samples))
            li_data_samples = li_data_samples[:sp_len]
        elif li_len > sp_len:
            while len(sp_data_samples) < li_len:
                sp_data_samples += random.sample(sp_data_samples, len(sp_data_samples))
            sp_data_samples = sp_data_samples[:li_len]


        li_data_samples = sorted(li_data_samples)
        sp_data_samples = sorted(sp_data_samples)

        random.shuffle(li_data_samples)
        random.shuffle(sp_data_samples)

        # tf.data.Dataset.from_tensor_slices()函数进行加载数据，把数据和标签一一对应
        dataset = tf.data.Dataset.from_tensor_slices((li_data_samples, sp_data_samples))
        # dataset = dataset.shuffle(shuffle_buffer_size).repeat(-1)  # 打乱数据（有规则的）
        # 程序这里是先shuffle再batch
        if train_mode == 'train':

            filepath = '/data/data5/yi/ECCV/csv/' + self.train_name +'-train.csv'

            write_in_csv2(li_data_samples, sp_data_samples, filepath)
        else:
            filepath = '/data/data5/yi/ECCV/csv/' + self.train_name +'-val.csv'

            write_in_csv2(li_data_samples, sp_data_samples, filepath)

        if train_mode == 'train':
            dataset = dataset.map(map_func=self.parse_fn, num_parallel_calls=autotune).repeat()
            dataset = dataset.batch(batch_size=self.config.BATCH_SIZE).prefetch(buffer_size=autotune)
        else:
            dataset = dataset.map(map_func=self.parse_fn_val, num_parallel_calls=autotune).repeat()
            # 每次输入一批batch_size个元素（这里n=2);一共buffer_size批次
            dataset = dataset.batch(batch_size=self.config.BATCH_SIZE).prefetch(buffer_size=autotune)  # prefetch并行化策略
        return dataset

    def inputs_for_testing(self):
        data_dir = self.config.LI_DATA_DIR + self.config.SP_DATA_DIR
        data_samples = []
        for _dir in data_dir:
            _list = sorted(glob.glob(_dir))
            data_samples += _list

        def list_extend(vd_list):
            new_list = []
            for _file in vd_list:
                meta = glob.glob(_file)  # .png——.tif
                new_list += meta
            return new_list

        data_samples = list_extend(data_samples)
        dataset = tf.data.Dataset.from_tensor_slices((data_samples))
        dataset = dataset.map(map_func=self.parse_fn_test, num_parallel_calls=autotune)
        dataset = dataset.batch(batch_size=self.config.BATCH_SIZE).prefetch(buffer_size=autotune)
        return dataset, data_samples

    def parse_fn(self, file1, file2):
        config = self.config
        imsize = config.IMAGE_SIZE

        def _parse_function(_file1, _file2):
            # live
            global fd
            _file1 = _file1.decode('UTF-8')
            meta = glob.glob(_file1)
            try:
                fd = meta[random.randint(0, len(meta) - 1)]
            except:
                print(_file1, len(meta))
            im_name = fd
            # lm_name = fd[:-3] + 'npy'

            fp = open(im_name, 'rb')
            image = Image.open(fp)

            width, height = image.size
            image_li = image.resize((imsize, imsize))
            fp.close()
            image_li = np.array(image_li, np.float32)
            if np.random.rand() > 0.5:
                image_li = cv2.flip(image_li, 1)


            # spoof
            _file2 = _file2.decode('UTF-8')
            meta = glob.glob(_file2)
            try:
                fd = meta[random.randint(0, len(meta) - 1)]
            except:
                print(_file2, len(meta))
            im_name = fd
            # lm_name = fd[:-3] + 'npy'
            fp = open(im_name, 'rb')
            image = Image.open(fp)

            width, height = image.size
            image_sp = image.resize((imsize, imsize))
            fp.close()
            image_sp = np.array(image_sp, np.float32)
            if np.random.rand() > 0.5:
                image_sp = cv2.flip(image_sp, 1)

            return np.array(image_li, np.float32) / 255, np.array(image_sp, np.float32) / 255

        image_li, image_sp = tf.py_func(_parse_function, [file1, file2],[tf.float32, tf.float32])  # 3个tf.float32
        # , reg_map_sp
        image_li = tf.ensure_shape(image_li, [imsize, imsize, 3])
        image_sp = tf.ensure_shape(image_sp, [imsize, imsize, 3])

        # data augmentation
        image = tf.stack([tf.image.random_brightness(image_li, 0.25), tf.image.random_brightness(image_sp, 0.25)],
                         axis=0)
        return image

    def parse_fn_val(self, file1, file2):
        config = self.config
        imsize = config.IMAGE_SIZE


        def _parse_function(_file1, _file2):
            # live

            global fd
            _file1 = _file1.decode('UTF-8')
            meta = glob.glob(_file1)
            try:
                fd = meta[random.randint(0, len(meta) - 1)]
            except:
                print(_file1, len(meta))
                input()
            im_name = fd
            # lm_name = fd[:-3] + 'npy'
            fp = open(im_name, 'rb')
            image = Image.open(fp)
            width, height = image.size
            image_li = image.resize((imsize, imsize))
            fp.close()
            image_li = np.array(image_li, np.float32)
            #if np.random.rand() > 0.5:
            #    image_li = cv2.flip(image_li, 1)

            # spoof
            _file2 = _file2.decode('UTF-8')
            meta = glob.glob(_file2)
            try:
                fd = meta[random.randint(0, len(meta) - 1)]
            except:
                print(_file2, len(meta))
                input()
            im_name = fd
            # lm_name = fd[:-3] + 'npy'
            fp = open(im_name, 'rb')
            image = Image.open(fp)
            width, height = image.size
            image_sp = image.resize((imsize, imsize))
            fp.close()
            image_sp = np.array(image_sp, np.float32)
            #if np.random.rand() > 0.5:
            #    image_sp = cv2.flip(image_sp, 1)

            return np.array(image_li, np.float32) / 255, np.array(image_sp, np.float32) / 255

        # tf.py_func:把tensorflow和python原生代码无缝衔接起来的函数
        image_li, image_sp = tf.py_func(_parse_function, [file1, file2],[tf.float32, tf.float32])
        # , reg_map_sp
        image_li = tf.ensure_shape(image_li, [imsize, imsize, 3])
        image_sp = tf.ensure_shape(image_sp, [imsize, imsize, 3])
        # reg_map_sp = tf.ensure_shape(reg_map_sp, [imsize, imsize, 3])

        # data augmentation[数据拼接]
        image = tf.stack([image_li, image_sp], axis=0)

        return image

    def parse_fn_test(self, file):
        config = self.config
        imsize = config.IMAGE_SIZE

        def _parse_function(_file):
            _file = _file.decode('UTF-8')
            image_list = []
            im_name = _file
            fp = open(im_name, 'rb')
            image = Image.open(fp)
            # image_ir = cv2.imread(im_name, 1)
            # image = Image.fromarray(cv2.cvtColor(image_ir, cv2.COLOR_BGR2RGB))
            # image = Image.open(im_name)
            image = image.resize((imsize, imsize))
            fp.close()
            return np.array(image, np.float32) / 255, im_name

        image, im_name = tf.py_func(_parse_function, [file], [tf.float32, tf.string])
        image = tf.ensure_shape(image, [config.IMAGE_SIZE, config.IMAGE_SIZE, 3])
        return image, im_name
