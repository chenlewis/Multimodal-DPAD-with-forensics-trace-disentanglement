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
import glob
import os
import time

import tensorflow as tf


# Base Configuration Class
class Config(object):
    """
    Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # GPU Usage
    GPU_USAGE = '0'

    # Log and Model Storage Default
    # LOG_DIR = 'D:/github/ECCV20-STDN-master/test_log/'
    #LOG_DIR = '/data/data2/ECCV20-STDN-master/stdn_log/'
    LOG_DIR = '/data/data5/yi/ECCV/stdn_log/'
    # checkpoint = 'D:/github/ECCV20-STDN-master/checkpoint/20230711102151/'
    #checkpoint = '/data/data2/ECCV20-STDN-master/checkpoint/20230713103633/'
    checkpoint = '/data/data5/yi/ECCV/checkpoint/20231121150107/'

    pretrained_ckpt_path = None

    LOG_DEVICE_PLACEMENT = None
    train_name = None

    # added losses参数
    epsilon = 1e-8
    ls_beta = 10
    lt_beta = 10
    lc_beta = 10

    # Input Data Meta
    IMAGE_SIZE = 224  # 256
    MAP_SIZE = 32

    # Training Meta
    BATCH_SIZE = 1
    G_D_RATIO = 2
    LEARNING_RATE = 2e-5# me:2e-5  # -5 6e-7
    LEARNING_RATE_DECAY_FACTOR = 0.9
    LEARNING_MOMENTUM = 0.999
    MAX_EPOCH = 50
    MOVING_AVERAGE_DECAY = 0.9999
    NUM_EPOCHS_PER_DECAY = 10.0
    STEPS_PER_EPOCH = 2000
    STEPS_PER_EPOCH_VAL = 500
    LOG_FR_TRAIN = int(STEPS_PER_EPOCH / 25)
    LOG_FR_TEST = int(STEPS_PER_EPOCH_VAL / 50)

    def __init__(self, gpu, root_dir, root_dir_val, mode):
        """Set values of computed attributes."""
        self.MODE = mode
        self.GPU_USAGE = gpu
        self.GPU_OPTIONS = tf.GPUOptions(per_process_gpu_memory_fraction=1, visible_device_list=self.GPU_USAGE,
                                         allow_growth=True)
        self.GPU_CONFIG = tf.ConfigProto(log_device_placement=self.LOG_DEVICE_PLACEMENT, gpu_options=self.GPU_OPTIONS)
        self.LI_DATA_DIR = [root_dir + 'live/*']
        self.SP_DATA_DIR = [root_dir + 'real_spoof/*']
        # self.ST_DATA_DIR = [root_dir + 'T/*']
        # self.SC_DATA_DIR = [root_dir + 'C/*']
        # self.RS_DATA_DIR = [root_dir + 'real_spoof/*']
        # self.TC_DATA_DIR = ['D:/DatabasesYI/TC_TEST_1/' + 'TC/*']
        if root_dir_val:
            self.LI_DATA_DIR_VAL = [root_dir_val + 'live/*']
            self.SP_DATA_DIR_VAL = [root_dir_val + 'real_spoof/*']
            # self.ST_DATA_DIR_VAL = [root_dir_val + 'T/*']
            # self.SC_DATA_DIR_VAL = [root_dir_val + 'C/*']
            # self.RS_DATA_DIR_VAL = [root_dir_val + 'real_spoof/*']
        self.compile()

    def compile(self):
        if not os.path.isdir(self.LOG_DIR):
            os.mkdir(self.LOG_DIR)
        if not os.path.isdir(self.LOG_DIR + '/test'):
            os.mkdir(self.LOG_DIR + '/test')
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)) and a[0].isupper():
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
