import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, cross_val_predict, ShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn import datasets
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import os
import time
from scipy.fftpack import fft,ifft
import itertools
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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    #clear diagonal values
    for index in range(len(classes)):
        cm[index][index] = 0

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

#################data upload

suffixes = ['hbj/', 'lyq/', 'yzc/', 'yyk/', 'yyw/', 'swn/', 'sy/', 'lgh/', 'ycy/']

accuracy_score_set = []

type_array = []
motion_type = []

for suffix in suffixes:
    rootdir = 'training/sound_final/'+suffix+'combination/'
    list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
    for i in range(0, len(list)):
        if (not 'swipe' in list[i]) and ('6' in list[i] or '3' in list[i] or '2' in list[i]):
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
    for j in range(len(type_array[i])):
        primitive_data = type_array[i][j]
        data_length = primitive_data.shape[0]
        bound = 26
        feature_length = 80
        featured_data = np.zeros((data_length, feature_length))
        print('type:', motion_type[i])
        for j in range(data_length):
            segment = primitive_data[j]
            featured_data[j] = feature_extraction_new(segment)

        print(featured_data.shape)

        feature_array[i].append(featured_data)

print(np.array(feature_array).shape)

##################display
# fig, axs = plt.subplots(len(feature_array), 1)
# index = 0
# for featured_data in feature_array:
#     for data_unit in featured_data:
#         for segment in data_unit:
#             axs[index].plot(segment)
#     index = index + 1
# plt.show()

###################label process
type_flag = []
for i in range(len(type_array)):
    type_flag.append([])
    for j in range(len(type_array[i])):
        flag = np.ones((type_array[i][j].shape[0])) * i
        type_flag[i].append(flag)
print(np.array(type_flag).shape)

###################calculate confusion

for i in range(len(suffixes)):
    res_matrix = np.zeros((len(motion_type), len(motion_type)), dtype=np.int)
    
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

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(predict_flag_set, y_pred)
    res_matrix = res_matrix + cnf_matrix
    np.set_printoptions(precision=2)
    class_names = motion_type

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(res_matrix, classes=class_names,
                        title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
    #                       title='Normalized confusion matrix')

    plt.show()

