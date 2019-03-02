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


type_array = []

motion_type = []

rootdir = 'training/motion/'
list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
for i in range(0, len(list)):
    motion_type.append(list[i])
    path = os.path.join(rootdir,list[i])
    print(path)
    data = np.load(path)
    print(data.shape)
    type_array.append(data)
print(len(type_array))

rootdir = 'training/motion_lu/'
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
print(len(type_array))

feature_array = []

for i in range(len(type_array)):
    primitive_data = type_array[i]
    data_length = primitive_data.shape[0]
    feature_length = 72 #72 features: (left, right) * (acc, att, rot) * (x, y, z) * (min, max, mean, std)
    # + (left, right) * (acc) * (x, y, z) * (frequency[5:25])
    featured_data = np.zeros((data_length, feature_length))
    for j in range(data_length):
        data_unit = primitive_data[j].reshape(50, 18)
        for k in range(18):
            # if not isRot(k):
            data_unit_coor = data_unit[:, k]
            featured_data[j, 4*k] = (int(np.min(data_unit_coor) * 1000)) / 1000
            featured_data[j, 4*k+1] = (int(np.max(data_unit_coor) * 1000)) / 1000
            featured_data[j, 4*k+2] = (int(np.mean(data_unit_coor) * 1000)) / 1000
            featured_data[j, 4*k+3] = (int(np.std(data_unit_coor) * 1000)) / 1000
        # for k in range(3):
        #     col = data_unit[:, k]
        #     col_fft = abs(fft(col))
        #     # print(col_fft.shape)
        #     featured_data[j, 48+k*10:48+(k+1)*10] = col_fft[15:25]
        #     col = data_unit[:, 9+k]
        #     col_fft = abs(fft(col))
        #     featured_data[j, 78+k*10:78+(k+1)*10] = col_fft[15:25]

    feature_array.append(featured_data)


print(len(feature_array))


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
cv =ShuffleSplit(10, test_size=0.2, train_size=0.8, random_state=seed)
scores = cross_validate(clf, feature_set, flag_set, cv=cv, return_train_score=True, return_estimator=True)
print(scores['test_score'])
print('max min mean = :', max(scores['test_score']), min(scores['test_score']), np.mean(scores['test_score']))
# print(min(scores), max(scores), np.mean(scores))
joblib.dump(scores['estimator'][np.argmax(scores['test_score'])], "model/classification24_withnoise_model.m")

'''
motion = np.load('training/palm_np.npy')
# print(np.shape(motion))
noise1 = np.load('training/noise_np.npy')
# print(np.shape(noise1))
noise2 = np.load('training/noise_16_np.npy')
# print(np.shape(noise2))
noise = np.concatenate((noise1, noise2), axis=0)
# print(np.shape(noise))

# concate
motion_flag = np.linspace(0, 0, motion.shape[0])
noise_flag = np.linspace(1, 1, noise.shape[0])

combine_set = np.concatenate((motion, noise))
print('set size: ', combine_set.shape)
flag_set = np.concatenate((motion_flag, noise_flag))
# print(flag_set.shape)

# scoring = ['precision_macro', 'recall_macro']
scoring = 'f1_macro'
clf = SVC(kernel='rbf', gamma='auto')# ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
scores = cross_validate(clf, combine_set, flag_set, scoring=scoring, cv=5, return_train_score=False, return_estimator=True)
print(scores.keys())
print(scores['test_score'])
print(scores['estimator'])

#joblib.dump(clf, "train_model.m")
#clf = joblib.load("train_model.m")
for i in range(5):
    joblib.dump(scores['estimator'][i], "model/palm_noise_svm_"+str(i)+".m")

# (1270, 900)
motion_display = motion.reshape(-1, 18)
length = (np.shape(motion_display))[0] / 50
index_array = np.arange(length)
#mix two datagram: original data visualization
fig, axs = plt.subplots(9, 1)
for i in range(9):
    axs[i].plot(index_array, motion_display[24::50,i], index_array, motion_display[24::50,i+9])
plt.show()
'''
