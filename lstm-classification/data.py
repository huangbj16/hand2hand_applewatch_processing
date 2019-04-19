import torch
import glob
import unicodedata
import string
import numpy as np
import os

def findFiles(path): return glob.glob(path)

# Read a file and split into lines
def read_data(filename):
    data = np.load(filename)
    print(data.shape)
    return data

# Build the category_lines dictionary, a list of lines per category
category_lines = []
all_categories = []
suffixes = ['hbj/', 'lyq/', 'yzc/', 'rj/']
for suffix in suffixes:
    rootdir = '../training/sound_final/'+suffix+'combination/'
    list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
    for i in range(0, len(list)):
        category = list[i].split('_')[0]
        all_categories.append(category)
        filename = os.path.join(rootdir,list[i])
        data = read_data(filename)
        print(data.shape)
        data = data[:, 0:1000]
        category_lines.append(data) #此时数据data为n*1000的nparray
    

n_categories = int(len(all_categories) / len(suffixes))
print(n_categories, all_categories)

flag_lines = []
for i in range(len(all_categories)):
    flag = np.ones(category_lines[i].shape[0]) * int(all_categories[i])
    flag_lines.append(flag)

feature_train_set = np.concatenate((category_lines[10:40]))
flag_train_set = np.concatenate((flag_lines[10:40]))
feature_test_set = np.concatenate((category_lines[0:10]))
flag_test_set = np.concatenate((flag_lines[0:10]))
print(feature_train_set.shape, flag_train_set.shape, feature_test_set.shape, flag_test_set.shape)