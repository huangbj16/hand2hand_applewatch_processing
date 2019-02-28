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

def isAcc(k):
    if k >= 6 and k < 9:
        return True
    elif k >= 15 and k < 18:
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

clf = SVC(kernel='rbf', gamma='auto')# ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
print('current time: ', time.time())
seed = int(time.time()*10000000) % 19980608
# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(feature_set, flag_set, test_size = 0.2, random_state=seed)


# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
y_pred = clf.fit(X_train, y_train).predict(X_test)


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
    for index in classes:
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


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

class_names = np.arange(24)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix')

plt.show()
