import csv
import os

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import glob


# filePath = 'D:/DatabasesYI/datasets_docu/test/test.csv'

def write_in_csv(list1, list2, list3, list4, filePath):
    rows = zip(list1, list2, list3, list4)
    header = ['live', 'spoof', 'T', 'C']
    with open(filePath, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def write_in_csv2(list1, list2, filePath):
    rows = zip(list1, list2)
    header = ['live', 'spoof']
    with open(filePath, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def write_in_csv5(list1, list2, list3, list4, list5, filePath):
    rows = zip(list1, list2, list3, list4, list5)
    header = ['live', 'svpervision', 'T', 'C', 'spoof']
    with open(filePath, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def write_in_csv7(list1, list2, list3, list4, list5, list6, list7, filePath):
    rows = zip(list1, list2, list3, list4, list5, list6, list7)
    header = ['live', 'svpervision', 'T', 'C', 'spoof', 'TLIVE', 'CLIVE']
    with open(filePath, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)



def shuffle_data(filepath, savepath):  # 按行打乱
    data = pd.read_csv(filepath)
    data = shuffle(data)
    data.to_csv(savepath)


def data_in_csv(filepath1):
    img_path = os.listdir(filepath1)
    src = []
    for img_name in img_path:
        src = src + list(os.path.join(filepath1, img_name))
    return src

