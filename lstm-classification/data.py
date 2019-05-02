import torch
import glob
import unicodedata
import string
import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft

def findFiles(path): return glob.glob(path)

# Read a file and split into lines
def read_data(filename):
    data = np.load(filename)
#     print(data.shape)
    return data

# Build the category_lines dictionary, a list of lines per category
category_lines = []
all_categories = []
# suffixes = ['sy/', 'hbj/', 'lyq/', 'yzc/', 'yyk/', 'yyw/', 'swn/', 'lgh/', 'ycy/']
suffixes = ['sy/', 'hbj/', 'lyq/']
excludes = ['6_np.npy', '4_np.npy']

for suffix in suffixes:
    rootdir = '../training/sound_final/'+suffix+'combination/'
    list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
    for i in range(0, len(list)):
        if list[i] in excludes or 'swipe' in list[i]:
            continue
        print(list[i])
        category = list[i].split('_')[0]
        all_categories.append(category)
        filename = os.path.join(rootdir,list[i])
        data = read_data(filename)
        audio_freq_left = abs(fft(data[:, 1000:1000+22050], axis=1))
        audio_freq_right = abs(fft(data[:, 1000+22050:1000+44100], axis=1))
        concat_data = np.concatenate((data[:, 0:1000], audio_freq_left[:, 0:1000], audio_freq_right[:, 0:1000]), axis=1)
        category_lines.append(concat_data) #此时数据data为n*1000的nparray

# pca_before = np.concatenate((category_lines))
# print(pca_before.shape)
# pca = PCA(n_components=100)
# pca_after =  pca.fit_transform(pca_before)
# print(pca_after.shape)

n_categories = int(len(all_categories) / len(suffixes))
print(n_categories, len(all_categories), all_categories)

flag_lines = []
for i in range(len(all_categories)):
    flag = np.ones(category_lines[i].shape[0]) * (i%n_categories)
    flag_lines.append(flag)

feature_set = np.concatenate((category_lines))
flag_set = np.concatenate((flag_lines))

test_length = np.sum([category_lines[i].shape[0] for i in range(n_categories)])
print(test_length)

feature_train_set = feature_set[test_length:]
flag_train_set = flag_set[test_length:]
feature_test_set = feature_set[:test_length] 
flag_test_set = flag_set[:test_length]
print(feature_train_set.shape, flag_train_set.shape, feature_test_set.shape, flag_test_set.shape)