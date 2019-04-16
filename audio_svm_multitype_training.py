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
type_array = []

motion_type = []

suffixes = ['hbj/', 'lyq/', 'yzc/']

for suffix in suffixes:
    rootdir = 'training/sound_final/'+suffix+'combination/'
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
    feature_length = 80+52
    featured_data = np.zeros((data_length, feature_length))
    print('type:', motion_type[i])
    for j in range(data_length):
        segment = primitive_data[j]
        featured_data[j] = feature_extraction_new(segment)
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

