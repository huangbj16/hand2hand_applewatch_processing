import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import datasets
from sklearn.externals import joblib
import matplotlib.pyplot as plt

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

# print(scores['test_precision_macro'])
# print(scores['test_recall_macro'])

#train
# X_train, X_test, y_train, y_test = train_test_split(
# ...     iris.data, iris.target, test_size=0.4, random_state=0)
# X_train, X_test, y_train, y_test = train_test_split(combine_set, flag_set, test_size = 0.2, random_state=0)
# print('train and test size: ', X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# clf = SVC(kernel='linear', C=1).fit(X_train, y_train)
# clf.score(X_test, y_test)
# clf = SVC(kernel='linear', C=1)
# scores = cross_val_score(clf, combine_set, flag_set, cv=5)
# print(scores)
# iris = datasets.load_iris()
# X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
# clf = SVC(kernel='linear', C=1).fit(X_train, y_train)
# print(clf.score(X_test, y_test))

# motion_len = (np.shape(motion))[0]
# motion_train_len = int(motion_len * 0.8)
# motion_test_len = motion_len - motion_train_len
# noise_len = (np.shape(noise))[0]
# noise_train_len = int(noise_len * 0.8)
# noise_test_len = noise_len - noise_train_len

# motion_split = np.split(motion, [motion_train_len, motion_len])
# noise_split = np.split(noise, [noise_train_len, noise_len])

# train_flag_set1 = np.linspace(0, 0, motion_train_len)
# train_flag_set2 = np.linspace(1, 1, noise_train_len)
# train_flag_set = np.concatenate((train_flag_set1, train_flag_set2))
# test_flag_set1 = np.linspace(0, 0, motion_test_len)
# test_flag_set2 = np.linspace(1, 1, noise_test_len)
# test_flag_set = np.concatenate((test_flag_set1, test_flag_set2))
# print(np.shape(train_flag_set1), np.shape(train_flag_set2), np.shape(train_flag_set), np.shape(test_flag_set1), np.shape(test_flag_set2), np.shape(test_flag_set))

# print(motion_split)
# print(noise_split)

# motion_train = motion_split[0]
# print(np.shape(motion_train))
# motion_test = motion_split[1]
# print(np.shape(motion_test))
# noise_train = noise_split[0]
# print(np.shape(noise_train))
# noise_test = noise_split[1]
# print(np.shape(noise_test))

# train_set = np.concatenate((motion_train, noise_train))
# test_set = np.concatenate((motion_test, noise_test))
# print(np.shape(train_set))
# print(np.shape(test_set))

# clf = SVC(gamma='auto')
# scores = cross_val_score(clf, train_set, train_flag_set, cv=5)
# print(scores)

# clf = SVC(gamma='auto')
# clf.fit(train_set, train_flag_set)
# res = clf.predict(test_set)
# print(res)
# print(test_flag_set)


# p = precision_score(test_flag_set, res, average='binary')
# r = recall_score(test_flag_set, res, average='binary')
# f1score = f1_score(test_flag_set, res, average='binary')

# print(p)
# print(r)
# print(f1score)

# (1270, 900)
motion_display = motion.reshape(-1, 18)
length = (np.shape(motion_display))[0] / 50
index_array = np.arange(length)
#mix two datagram: original data visualization
fig, axs = plt.subplots(9, 1)
for i in range(9):
    axs[i].plot(index_array, motion_display[24::50,i], index_array, motion_display[24::50,i+9])
plt.show()
