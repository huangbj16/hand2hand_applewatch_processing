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
type_array = []

motion_type = []

suffixes = ['hbj/', 'lyq/', 'jzs/', 'ljh/']

for suffix in suffixes:
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
    feature_length = 52+72
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

        # from Detecting and Classifying Human Touches in a Social Robot Through Acoustic Sensing and Machine Learning
        # maximum, minimum, and average values of pitch, flux, roll-off, centroid, ZCR, RMS, and SNR.
        # basic features
        # root_mean_square = [np.sqrt(((audio_left).astype('double') ** 2).mean()), np.sqrt(((audio_right).astype('double') ** 2).mean())]
        # audio_left = audio_left / root_mean_square[0]
        # audio_right = audio_right / root_mean_square[1]
        # audio_max = [np.max(audio_left), np.max(audio_right)]
        # audio_min = [np.min(audio_left), np.min(audio_right)]
        # audio_mean = [np.mean(audio_left), np.mean(audio_right)]
        # audio_std = [np.std(audio_left), np.std(audio_right)]
        # featured_data[j] = np.concatenate((audio_max, audio_min, audio_mean, audio_std))


        # high-dimension features
        # root_mean_square = [np.sqrt(((audio_left).astype('double') ** 2).mean()), np.sqrt(((audio_right).astype('double') ** 2).mean())]
        # # lacking zero crossing rate, can't implemented.
        # audio_left = audio_left / root_mean_square[0]
        # audio_right = audio_right / root_mean_square[1]
        # freq_audio_left = np.array(abs(fft(audio_left)))
        # freq_audio_right = np.array(abs(fft(audio_right)))
        # freq_audio_left = freq_audio_left[:11025]
        # freq_audio_right = freq_audio_right[:11025]
        # freq_energy = [np.sum(freq_audio_left), np.sum(freq_audio_right)]
        # roll_off = [-1, -1]
        # tot = [0, 0]
        # for k in range(11025):
        #     tot[0] = tot[0] + freq_audio_left[k]
        #     tot[1] = tot[1] + freq_audio_right[k]
        #     if tot[0] > freq_energy[0] * 0.95 and roll_off[0] == -1:
        #         roll_off[0] = k
        #     if tot[1] > freq_energy[1] * 0.95 and roll_off[1] == -1:
        #         roll_off[1] = k
        # freq_index = np.arange(11025)
        # centroid = [np.sum(freq_audio_left*freq_index) / np.sum(freq_audio_left), np.sum(freq_audio_right*freq_index) / np.sum(freq_audio_right)]
        # featured_data[j] = np.concatenate((root_mean_square))

        # freq_audio_left = freq_audio_left[:5000]
        # freq_audio_right = freq_audio_right[:5000]


        ##############feature: stft
        ##############display
        # fs = 44100
        # f, t, Zxx = signal.stft(audio_left, fs, nperseg=2100)
        # Zxx = np.abs(Zxx)     
        
        # # print(f.shape, t.shape, Zxx.shape)
        # # print(f[:20], t[:20], Zxx[0][0])
        # # plt.pcolormesh(t, f[:20], np.abs(Zxx)[:20], vmin=np.min(audio_left), vmax=np.max(audio_left))
        # # plt.title('STFT Magnitude')
        # # plt.ylabel('Frequency [Hz]')
        # # plt.xlabel('Time [sec]')
        # # plt.show()

        # Zxx = np.sum(Zxx, axis=1)
        # Zxx = Zxx / np.linalg.norm(Zxx)
        # featured_data[j, 0:bound] = Zxx[:25]
        # f, t, Zxx = signal.stft(audio_right, fs, nperseg=2100)
        # Zxx = np.abs(Zxx)
        # Zxx = np.sum(Zxx, axis=1)
        # Zxx = Zxx / np.linalg.norm(Zxx)
        # featured_data[j, bound:] = Zxx[:25]
        

        ##############feature: peaks in audio freq
        # peaks_left, _ = find_peaks(freq_audio_left, distance=100, height=1)
        # peaks_right, _ = find_peaks(freq_audio_right, distance=100, height=1)        
        ##############display
        # if i != 1:
        #     break
        # print(peaks_left, peaks_right)
        # fig, axs = plt.subplots(2, 1)        
        # axs[0].plot(freq_audio_left)
        # axs[0].plot(peaks_left, freq_audio_left[peaks_left], "x")
        # axs[1].plot(freq_audio_right)
        # axs[1].plot(peaks_right, freq_audio_right[peaks_right], "x")
        # plt.show()
        # continue
        ##############complement and curtail to 10
        # comple_array = np.ones((10), dtype=np.int)*(-100)
        # if peaks_left.shape[0] < 10:
        #     peaks_left = np.append(peaks_left, comple_array)
        # if peaks_right.shape[0] < 10:
        #     peaks_right = np.append(peaks_right, comple_array)
        # featured_data[j, :10] = peaks_left[:10]
        # featured_data[j, 10:20] = peaks_right[:10]


        ##############feature: bucket max min mean = : 0.9057750759878419 0.8206686930091185 0.8540425531914891
        ######normalize 
        # freq_audio_left = freq_audio_left / np.linalg.norm(freq_audio_left)
        # freq_audio_right = freq_audio_right / np.linalg.norm(freq_audio_right)
        # bucket_index = [0, 100, 200, 400, 800, 1600, 2400, 3200, 4000, 6400, 11025]
        # for k in range(len(bucket_index)-1):
        #     lower_bound = bucket_index[k]
        #     upper_bound = bucket_index[k+1]
        #     featured_data[j, k] = np.sum(freq_audio_left[lower_bound:upper_bound])
        #     featured_data[j, 10+k] = np.sum(freq_audio_right[lower_bound:upper_bound])

        ##############feature: uniform bucket
        # freq_energy_audio_left = np.sum(freq_audio_left, axis=1)
        # freq_energy_audio_right = np.sum(freq_audio_right, axis=1)
        # print(freq_energy_audio_left.shape, freq_energy_audio_right.shape)
        # featured_data[j, 0:bound] = freq_energy_audio_left[0:bound]
        # featured_data[j, bound:2*bound] = freq_energy_audio_right[0:bound]
        

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

        ###############feature: brute force audio
        # featured_data[j] = segment[900:]
        
        
        ##############feature: audio time (min, max, mean, std): accuracy 0.95
        # featured_data[j, 0] = np.min(audio_left)
        # featured_data[j, 1] = np.max(audio_left)
        # featured_data[j, 2] = np.mean(audio_left)
        # featured_data[j, 3] = np.std(audio_left)
        # featured_data[j, 4] = np.min(audio_right)
        # featured_data[j, 5] = np.max(audio_right)
        # featured_data[j, 6] = np.mean(audio_right)
        # featured_data[j, 7] = np.std(audio_right)

    print(featured_data.shape)

    feature_array.append(featured_data)

print(len(feature_array))

# exit(0)

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

#concatenate
type_set = np.concatenate(type_array)
flag_set = np.concatenate(type_flag)
feature_set = np.concatenate(feature_array)
print(type_set.shape, flag_set.shape, feature_set.shape)

# scoring = ['precision_macro', 'recall_macro']
# scoring = 'f1_macro'
# clf = SVC(kernel='rbf', gamma='auto')# ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
# scores = cross_validate(clf, type_set, flag_set, scoring=scoring, cv=5, return_train_score=False, return_estimator=True)
# print(scores.keys())
# print(scores['test_score'])

print(feature_set)

clf = SVC(kernel='rbf', gamma='auto')# ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’

################use p1 data to predict p2 data
# X_train = np.concatenate((feature_set[0:46], feature_set[92:142], feature_set[192:275]))
# y_train = np.concatenate((flag_set[0:46], flag_set[92:142], flag_set[192:275]))
# X_test = np.concatenate((feature_set[46:92], feature_set[142:192], feature_set[275:325]))
# y_test = np.concatenate((flag_set[46:92], flag_set[142:192], flag_set[275:325]))
# y_pred = clf.fit(X_train, y_train).predict(X_test)
# print(accuracy_score(y_test, y_pred))
# exit(0)

print('current time: ', time.time())
seed = int(time.time()*10000000) % 19980608
cv =ShuffleSplit(100, test_size=0.2, train_size=0.8, random_state=seed)
scores = cross_validate(clf, feature_set, flag_set, cv=cv, return_train_score=True, return_estimator=True)
print(scores['test_score'])
print('max min mean = :', max(scores['test_score']), min(scores['test_score']), np.mean(scores['test_score']))
# print(min(scores), max(scores), np.mean(scores))
joblib.dump(scores['estimator'][np.argmax(scores['test_score'])], "model/classification_audio_model.m")

