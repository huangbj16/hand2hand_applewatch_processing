import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, cross_val_predict, ShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import datasets
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import os
import time
from scipy.fftpack import fft,ifft
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
from python_speech_features  import mfcc


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

rootdir = 'training/sound/hbj'
list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
for i in range(0, len(list)):
    motion_type.append(list[i])
    path = os.path.join(rootdir,list[i])
    print(path)
    data = np.load(path)
    print(data.shape)
    type_array.append(data)
print(len(type_array))


rootdir = 'training/sound/lyq'
list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
for i in range(0, len(list)):
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
print(len(type_array))
print(motion_type)

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
    bound = 20
    feature_length = 20
    featured_data = np.zeros((data_length, feature_length))
    for j in range(data_length):
        segment = primitive_data[j]
        data_unit = segment[0:900].reshape(50, 18)
        audio_left = segment[900:900+22050]
        audio_right = segment[900+22050:900+44100]
        freq_audio_left = np.array(abs(fft(audio_left)))[:5000]
        freq_audio_right = np.array(abs(fft(audio_right)))[:5000]

        ##############feature: peaks in audio freq
        peaks_left, _ = find_peaks(freq_audio_left, distance=100, height=5)
        peaks_right, _ = find_peaks(freq_audio_right, distance=100, height=5)        
        ##############display
        # if i == 0:
        #     continue
        # print(peaks_left, peaks_right)
        # fig, axs = plt.subplots(2, 1)        
        # axs[0].plot(freq_audio_left)
        # axs[0].plot(peaks_left, freq_audio_left[peaks_left], "x")
        # axs[1].plot(freq_audio_right)
        # axs[1].plot(peaks_right, freq_audio_right[peaks_right], "x")
        # plt.show()
        # continue
        ##############complement and curtail to 10
        comple_array = np.ones((10), dtype=np.int)*(-100)
        if peaks_left.shape[0] < 10:
            peaks_left = np.append(peaks_left, comple_array)
        if peaks_right.shape[0] < 10:
            peaks_right = np.append(peaks_right, comple_array)
        featured_data[j, :10] = peaks_left[:10]
        featured_data[j, 10:20] = peaks_right[:10]

        # freq_energy_audio_left = np.sum(freq_audio_left, axis=1)
        # freq_energy_audio_right = np.sum(freq_audio_right, axis=1)
        # print(freq_energy_audio_left.shape, freq_energy_audio_right.shape)

        # featured_data[j, 0:bound] = freq_energy_audio_left[0:bound]
        # featured_data[j, bound:2*bound] = freq_energy_audio_right[0:bound]
        
        # sampling_freq = 44100
        # fft_size = 22050
        # mfcc_left = mfcc(audio_left, samplerate=sampling_freq, winlen=0.25, winstep=0.125, nfft=fft_size)
        # mfcc_right = mfcc(audio_right, samplerate=sampling_freq, winlen=0.25, winstep=0.125, nfft=fft_size)
        # # print(mfcc_left.shape, mfcc_right.shape)
        # featured_data[j, 0:bound] = mfcc_left.reshape(-1)
        # featured_data[j, bound:2*bound] = mfcc_right.reshape(-1)

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
print('current time: ', time.time())
seed = int(time.time()*10000000) % 19980608
cv =ShuffleSplit(100, test_size=0.2, train_size=0.8, random_state=seed)
scores = cross_validate(clf, feature_set, flag_set, cv=cv, return_train_score=True, return_estimator=True)
print(scores['test_score'])
print('max min mean = :', max(scores['test_score']), min(scores['test_score']), np.mean(scores['test_score']))
# print(min(scores), max(scores), np.mean(scores))
joblib.dump(scores['estimator'][np.argmax(scores['test_score'])], "model/classification5_IyPsubset_model.m")

