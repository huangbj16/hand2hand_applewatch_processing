import torch
import glob
import unicodedata
import string
import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
from python_audio_feature import mfcc

def findFiles(path): return glob.glob(path)

# Read a file and split into lines
def read_data(filename):
    data = np.load(filename)
    # print(data.shape)
    return data

# Build the category_lines dictionary, a list of lines per category
category_lines = []
all_categories = []
suffixes = ['sy/', 'hbj/', 'lyq/', 'yzc/', 'yyk/', 'yyw/', 'swn/', 'lgh/', 'ycy/']
# suffixes = ['sy/', 'hbj/']
excludes = ['6_np.npy', '4_np.npy']

for suffix in suffixes:
    rootdir = '../training/temp/'+suffix
    list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
    for i in range(0, len(list)):
        if list[i] in excludes:
            continue
        print(list[i])
        category = list[i].split('_')[0]
        all_categories.append(category)
        filename = os.path.join(rootdir,list[i])
        data = read_data(filename)
        # audio_freq_left = (abs(fft(data[:, 1000:1000+22050], axis=1)))[:, 0:1000]
        # audio_freq_right = (abs(fft(data[:, 1000+22050:1000+44100], axis=1)))[:, 0:1000]
        # audio_freq_left = np.sum(audio_freq_left.reshape(-1, 100, 10), axis=2)
        # audio_freq_right = np.sum(audio_freq_right.reshape(-1, 100, 10), axis=2)
        
        ##mfcc
        # sampling_freq = 44100
        # fft_size = 22050
        # audio_left = data[:, 1000:1000+22050]
        # audio_right = data[:, 1000+22050:1000+44100]
        # concat_data = np.zeros((audio_left.shape[0], 52+1000))
        # for i in range(audio_left.shape[0]):
        #     audio_left[i] = audio_left[i] / np.linalg.norm(audio_left[i])
        #     audio_right[i] = audio_right[i] / np.linalg.norm(audio_right[i])
        #     mfcc_left = mfcc(audio_left[i], samplerate=sampling_freq, winlen=0.5, winstep=0.25, nfft=fft_size)
        #     mfcc_right = mfcc(audio_right[i], samplerate=sampling_freq, winlen=0.5, winstep=0.25, nfft=fft_size)
        #     concat_data[i][0:26] = mfcc_left
        #     concat_data[i][26:52] = mfcc_right
        #     concat_data[i][52:52+1000] = data[i][0:1000]
        #     # plt.plot(concat_data[i])

        # concat_data = np.concatenate((data[:, 0:1000], audio_freq_left, audio_freq_right), axis=1)
        # plt.show()
        category_lines.append(data[:, 0:1000]) #此时数据data为n*(52+1000)的nparray

pca_before = np.concatenate((category_lines))
# print(pca_before.shape)
# pca = PCA(n_components=100)
# pca_after =  pca.fit_transform(pca_before)
# print(pca_after.shape)
pca_after = pca_before

print(pca_after.shape)

n_categories = int(len(all_categories) / len(suffixes))
print(n_categories, len(all_categories), all_categories)

flag_lines = []
for i in range(len(all_categories)):
    flag = np.ones(category_lines[i].shape[0]) * (i%n_categories)
    flag_lines.append(flag)

feature_set = pca_after
flag_set = np.concatenate((flag_lines))
print(flag_set)