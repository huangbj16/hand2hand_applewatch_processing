import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, cross_val_predict, ShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn import datasets
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import os
import time
from scipy.fftpack import fft,ifft
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
from scipy import signal
import sys
sys.path.append('/')
from python_audio_feature import mfcc
from feature_extraction_module import feature_extraction_new, feature_extraction_old

def isRot(k):
    if k >= 6 and k < 9:
        return True
    elif k >= 15 and k < 18:
        return True
    else:
        return False

def isAcc(k):
    if k >= 0 and k < 3:
        return True
    elif k >= 9 and k < 12:
        return True
    else:
        return False

def isAtt(k):
    if k >= 3 and k < 6:
        return True
    elif k >= 12 and k < 15:
        return True
    else:
        return False

#################data upload

suffixes = ['hbj/', 'lyq/', 'yzc/', 'rj/']

accuracy_score_set = []

type_array = []
motion_type = []

for suffix in suffixes:
    rootdir = 'training/sound_final/'+suffix+'combination/'
    list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
    for i in range(0, len(list)):
        if '6' in list[i] or '4' in list[i]:
            continue
        path = os.path.join(rootdir,list[i])
        print(path)
        data = np.load(path)
        print(data.shape)
        if list[i] in motion_type:
            mark = motion_type.index(list[i])
            type_array[mark].append(data)
        else:
            type_array.append([data])
            motion_type.append(list[i])

print(len(motion_type), motion_type)

'''
type_array format = [[t1d1, t1d2, t1d3...], [t2d1, t2d2, t2d3...], [t3d1...]]
'''

#####################feature process
feature_array = []

for i in range(len(type_array)):
    feature_array.append([])
    print('feature of ', motion_type[i])
    for j in range(len(type_array[i])):
        print('user ', suffixes[j])
        primitive_data = type_array[i][j]
        data_length = primitive_data.shape[0]
        bound = 26
        feature_length = 80
        featured_data = np.zeros((data_length, feature_length))
        for j in range(data_length):
            segment = primitive_data[j]
            featured_data[j] = feature_extraction_new(segment)
        print(featured_data.shape)
        feature_array[i].append(featured_data)

print(np.array(feature_array).shape)

##################display
fig, axs = plt.subplots(len(suffixes), len(feature_array))
index = 0
for featured_data in feature_array:
    index_user = 0
    for data_unit in featured_data:
        for segment in data_unit:
            axs[index_user][index].plot(segment[0:20])
        index_user = index_user + 1
    index = index + 1
# plt.setp(axs, ylim=(-10, 10))
plt.show()

###################label process
type_flag = []
for i in range(len(type_array)):
    type_flag.append([])
    for j in range(len(type_array[i])):
        flag = np.ones((type_array[i][j].shape[0])) * i
        type_flag[i].append(flag)
print(np.array(type_flag).shape)

####################independent classification

for i in range(len(suffixes)):

    #####################display to see manually

    # for data in type_array:
    #     print(data.shape)
    #     for i in range(data.shape[0]):
    #         segment = data[i]
    #         data_unit = segment[0:900].reshape(50, 18)
    #         audio_left = segment[900:900+22050]
    #         audio_right = segment[900+22050:900+44100]
    #         fig, axs = plt.subplots(5, 1)
    #         for j in range(3):
    #             axs[j].plot(range(50), data_unit[:, j], range(50), data_unit[:, 9+j])
    #         axs[3].plot(audio_left)
    #         axs[4].plot(audio_right)
    #         plt.show()
    #         if i == 5:
    #             break        

    # exit(0)
  
    #concatenate
    flag_set = []
    predict_flag_set = []
    feature_set = []
    predict_feature_set = []
    for j in range(len(type_flag)):
        for k in range(len(type_flag[j])):
            if k == i:#p2
                predict_flag_set.append(type_flag[j][k])
                predict_feature_set.append(feature_array[j][k])
            else:
                flag_set.append(type_flag[j][k])
                feature_set.append(feature_array[j][k])
    predict_flag_set = np.concatenate(predict_flag_set)
    predict_feature_set = np.concatenate(predict_feature_set)
    flag_set = np.concatenate(flag_set)
    feature_set = np.concatenate(feature_set)
    clf = SVC(kernel='rbf', gamma='auto')# ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’

    ################use p1 data to predict p2 data
    y_pred = clf.fit(feature_set, flag_set).predict(predict_feature_set)
    print(accuracy_score(predict_flag_set, y_pred))
    accuracy_score_set.append(accuracy_score(predict_flag_set, y_pred))

print(accuracy_score_set)

