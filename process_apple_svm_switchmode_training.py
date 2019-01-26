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

def isAcc(k):
    if k >= 6 and k < 9:
        return True
    elif k >= 15 and k < 18:
        return True
    else:
        return False

type_array = []

rootdir = 'D:/2018autumn/hand2hand/hand2hand_applewatch_processing/training/motion/'
list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
for i in range(0,len(list)):
    path = os.path.join(rootdir,list[i])
    print(path)
    data = np.load(path)
    print(data.shape)
    type_array.append(data)
print(len(type_array))

feature_array = []

for i in range(len(type_array)):
    primitive_data = type_array[i]
    data_length = primitive_data.shape[0]
    feature_length = 48 #48 features: (left, right) * (acc, att) * (x, y, z) * (min, max, mean, std)
    # + (left, right) * (acc) * (x, y, z) * (frequency[5:25])
    featured_data = np.zeros((data_length, feature_length))
    for j in range(data_length):
        data_unit = primitive_data[j].reshape(50, 18)
        for k in range(18):
            if not isAcc(k):
                data_unit_coor = data_unit[:, k]
                if k >= 9:
                    k = k - 3
                featured_data[j, 4*k] = np.min(data_unit_coor)
                featured_data[j, 4*k+1] = np.max(data_unit_coor)
                featured_data[j, 4*k+2] = np.mean(data_unit_coor)
                featured_data[j, 4*k+3] = np.std(data_unit_coor)
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

#concatenate
type_set = np.concatenate(type_array)
# flag_set = np.concatenate(type_flag)
feature_set = np.concatenate(feature_array)

for i in range(len(type_array)):
    print('current signal: ', i)
    type_flag = []
    for j in range(len(type_array)):
        if i == j:
            flag = np.ones((type_array[j].shape[0]))
        else:
            flag = np.zeros((type_array[j].shape[0]))
        # print(flag)
        type_flag.append(flag)
    
    flag_set = np.concatenate(type_flag)
    clf = SVC(kernel='rbf', gamma='auto')# ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
    # print(time.time())
    seed = int(time.time()*10000000) % 19980608
    cv =ShuffleSplit(10, test_size=0.2, train_size=0.8, random_state=seed)
    scores = cross_validate(clf, feature_set, flag_set, cv=cv, return_train_score=True, return_estimator=True)
    # print(scores['test_score'])
    print('max min mean = :', max(scores['test_score']), min(scores['test_score']), np.mean(scores['test_score']))
    # print(min(scores), max(scores), np.mean(scores))
    # joblib.dump(scores['estimator'][np.argmax(scores['test_score'])], "model/classification10_withnoise_model.m")

# scoring = ['precision_macro', 'recall_macro']
# scoring = 'f1_macro'
# clf = SVC(kernel='rbf', gamma='auto')# ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
# scores = cross_validate(clf, type_set, flag_set, scoring=scoring, cv=5, return_train_score=False, return_estimator=True)
# print(scores.keys())
# print(scores['test_score'])
# print(feature_set[0])
