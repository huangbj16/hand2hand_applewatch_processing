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

suffixes = ['hbj/', 'lyq/', 'jzs' ,'ljh/']

for predict_suffix in suffixes:

    type_array = []
    motion_type = []

    for suffix in suffixes:
        if suffix == predict_suffix:
            continue
        rootdir = 'training/sound/'+suffix
        list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
        for i in range(0, len(list)):
            if 'IyP' in list[i]:
                continue
            path = os.path.join(rootdir,list[i])
            print(path)
            data = np.load(path)
            print(data.shape)
            if list[i] in motion_type:
                mark = motion_type.index(list[i])
                previous_data = type_array[mark]
                type_array[mark] = np.concatenate((previous_data, data))
            else:
                type_array.append(data)
                motion_type.append(list[i])

    print(len(motion_type), motion_type)

    predict_type_array = []
    predict_motion_type = []
    rootdir = 'training/sound/'+predict_suffix
    list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
    for i in range(0, len(list)):
        if 'IyP' in list[i]:
            continue
        path = os.path.join(rootdir,list[i])
        print(path)
        data = np.load(path)
        print(data.shape)
        if list[i] in predict_motion_type:
            mark = predict_motion_type.index(list[i])
            previous_data = predict_type_array[mark]
            predict_type_array[mark] = np.concatenate((previous_data, data))
        else:
            predict_type_array.append(data)
            predict_motion_type.append(list[i])

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

    #####################feature process
    feature_array = []

    for i in range(len(type_array)):
        primitive_data = type_array[i]
        data_length = primitive_data.shape[0]
        bound = 26
        feature_length = 72+52
        featured_data = np.zeros((data_length, feature_length))
        print('type:', motion_type[i])
        for j in range(data_length):
            segment = primitive_data[j]
            data_unit = segment[0:900].reshape(50, 18)
            audio_left = segment[900:900+22050]
            audio_right = segment[900+22050:900+44100]
            freq_audio_left = np.array(abs(fft(audio_left)))
            freq_audio_right = np.array(abs(fft(audio_right)))

            feature_offset = 52
            for k in range(18):
                # if not isRot(k):
                data_unit_coor = data_unit[:, k]
                featured_data[j, feature_offset + 4*k] = (int(np.min(data_unit_coor) * 1000)) / 1000
                featured_data[j, feature_offset + 4*k+1] = (int(np.max(data_unit_coor) * 1000)) / 1000
                featured_data[j, feature_offset + 4*k+2] = (int(np.mean(data_unit_coor) * 1000)) / 1000
                featured_data[j, feature_offset + 4*k+3] = (int(np.std(data_unit_coor) * 1000)) / 1000

            ##############feature: mfcc max min mean = : 0.9513677811550152 0.8844984802431611 0.9130699088145896
            sampling_freq = 44100
            fft_size = 22050
            audio_left = audio_left / np.linalg.norm(audio_left)
            audio_right = audio_right / np.linalg.norm(audio_right)
            mfcc_left = mfcc(audio_left, samplerate=sampling_freq, winlen=0.5, winstep=0.25, nfft=fft_size)
            mfcc_right = mfcc(audio_right, samplerate=sampling_freq, winlen=0.5, winstep=0.25, nfft=fft_size)
            # print(mfcc_left.shape, mfcc_right.shape)
            # print(np.mean(mfcc_left, axis=0).shape)
            # exit(0)
            featured_data[j, 0:bound] = np.abs(mfcc_left).reshape(-1)
            featured_data[j, bound:2*bound] = np.abs(mfcc_right).reshape(-1)
            # featured_data[j, 0:bound] = np.amax(mfcc_left, axis=0)
            # featured_data[j, bound:2*bound] = np.amax(mfcc_right, axis=0)

        print(featured_data.shape)

        feature_array.append(featured_data)

    print(len(feature_array))

    # exit(0)

    predict_feature_array = []

    for i in range(len(predict_type_array)):
        primitive_data = predict_type_array[i]
        data_length = primitive_data.shape[0]
        bound = 26
        feature_length = 72+52
        featured_data = np.zeros((data_length, feature_length))
        print('type:', motion_type[i])
        for j in range(data_length):
            segment = primitive_data[j]
            data_unit = segment[0:900].reshape(50, 18)
            audio_left = segment[900:900+22050]
            audio_right = segment[900+22050:900+44100]
            freq_audio_left = np.array(abs(fft(audio_left)))
            freq_audio_right = np.array(abs(fft(audio_right)))

            feature_offset = 52
            for k in range(18):
                # if not isRot(k):
                data_unit_coor = data_unit[:, k]
                featured_data[j, feature_offset + 4*k] = (int(np.min(data_unit_coor) * 1000)) / 1000
                featured_data[j, feature_offset + 4*k+1] = (int(np.max(data_unit_coor) * 1000)) / 1000
                featured_data[j, feature_offset + 4*k+2] = (int(np.mean(data_unit_coor) * 1000)) / 1000
                featured_data[j, feature_offset + 4*k+3] = (int(np.std(data_unit_coor) * 1000)) / 1000

            ##############feature: mfcc max min mean = : 0.9513677811550152 0.8844984802431611 0.9130699088145896
            sampling_freq = 44100
            fft_size = 22050
            audio_left = audio_left / np.linalg.norm(audio_left)
            audio_right = audio_right / np.linalg.norm(audio_right)
            mfcc_left = mfcc(audio_left, samplerate=sampling_freq, winlen=0.5, winstep=0.25, nfft=fft_size)
            mfcc_right = mfcc(audio_right, samplerate=sampling_freq, winlen=0.5, winstep=0.25, nfft=fft_size)
            # print(mfcc_left.shape, mfcc_right.shape)
            # print(np.mean(mfcc_left, axis=0).shape)
            # exit(0)
            featured_data[j, 0:bound] = np.abs(mfcc_left).reshape(-1)
            featured_data[j, bound:2*bound] = np.abs(mfcc_right).reshape(-1)
            # featured_data[j, 0:bound] = np.amax(mfcc_left, axis=0)
            # featured_data[j, bound:2*bound] = np.amax(mfcc_right, axis=0)

        print(featured_data.shape)

        predict_feature_array.append(featured_data)

    print(len(predict_feature_array))

    ##################display
    fig, axs = plt.subplots(len(feature_array), 1)
    index = 0
    for featured_data in feature_array:
        print(featured_data.shape)
        # fig, axs = plt.subplots(13, 2) 
        # for i in range(featured_data.shape[1]):
        #     axs[int(i/2)][int(i%2)].plot(featured_data[:, i])
        # plt.show()
        # fig, axs = plt.subplots(5, 1)
        for segment in featured_data:
            axs[index].plot(segment)
        index = index + 1
    plt.show()

    ###################label process
    type_flag = []
    for i in range(len(type_array)):
        flag = np.ones((type_array[i].shape[0])) * i
        print(flag)
        type_flag.append(flag)

    predict_type_flag = []
    for i in range(len(predict_type_array)):
        flag = np.ones((predict_type_array[i].shape[0])) * i
        print(flag)
        predict_type_flag.append(flag)    

    #concatenate
    type_set = np.concatenate(type_array)
    flag_set = np.concatenate(type_flag)
    feature_set = np.concatenate(feature_array)
    print(type_set.shape, flag_set.shape, feature_set.shape)
    predict_type_set = np.concatenate(predict_type_array)
    predict_flag_set = np.concatenate(predict_type_flag)
    predict_feature_set = np.concatenate(predict_feature_array)
    print(predict_type_set.shape, predict_flag_set.shape, predict_feature_set.shape)

    clf = SVC(kernel='rbf', gamma='auto')# ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’

    ################use p1 data to predict p2 data
    y_pred = clf.fit(feature_set, flag_set).predict(predict_feature_set)
    print(accuracy_score(predict_flag_set, y_pred))

