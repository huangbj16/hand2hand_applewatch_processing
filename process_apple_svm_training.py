import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# f = open('svm_data.txt', 'r')
# lines = f.readlines()
# f.close()
# X = []
# y = []
# for line in lines:
#     conts = line.split()
#     motion_type = int(conts[0])
#     data = []
#     contslen = len(conts)
#     for i in range(1, contslen):
#         data.append(float(conts[i]))
#     y.append(motion_type)
#     X.append(data)
#
# # X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
# # y = np.array([1, 1, 2, 3])
# clf = SVC(gamma='auto')
# clf.fit(X, y)

motion = np.load('training/palm_np.npy')
print(np.shape(motion))
noise1 = np.load('training/noise_np.npy')
print(np.shape(noise1))
noise2 = np.load('training/noise_16_np.npy')
print(np.shape(noise2))
noise = np.concatenate((noise1, noise2), axis=0)
print(np.shape(noise))

motion_len = (np.shape(motion))[0]
motion_train_len = int(motion_len * 0.8)
motion_test_len = motion_len - motion_train_len
noise_len = (np.shape(noise))[0]
noise_train_len = int(noise_len * 0.8)
noise_test_len = noise_len - noise_train_len

motion_split = np.split(motion, [motion_train_len, motion_len])
noise_split = np.split(noise, [noise_train_len, noise_len])

train_flag_set1 = np.linspace(1, 1, motion_train_len)
train_flag_set2 = np.linspace(2, 2, noise_train_len)
train_flag_set = np.concatenate((train_flag_set1, train_flag_set2))
test_flag_set1 = np.linspace(1, 1, motion_test_len)
test_flag_set2 = np.linspace(2, 2, noise_test_len)
test_flag_set = np.concatenate((test_flag_set1, test_flag_set2))
print(np.shape(train_flag_set1), np.shape(train_flag_set2), np.shape(train_flag_set), np.shape(test_flag_set1), np.shape(test_flag_set2), np.shape(test_flag_set))

print(motion_split)
print(noise_split)

motion_train = motion_split[0]
print(np.shape(motion_train))
motion_test = motion_split[1]
print(np.shape(motion_test))
noise_train = noise_split[0]
print(np.shape(noise_train))
noise_test = noise_split[1]
print(np.shape(noise_test))

train_set = np.concatenate((motion_train, noise_train))
test_set = np.concatenate((motion_test, noise_test))
print(np.shape(train_set))
print(np.shape(test_set))

clf = SVC(gamma='auto')
clf.fit(train_set, train_flag_set)
res = clf.predict(test_set)
res = res - 1
test_flag_set = test_flag_set - 1
print(res)
print(test_flag_set)


p = precision_score(test_flag_set, res, average='binary')
r = recall_score(test_flag_set, res, average='binary')
f1score = f1_score(test_flag_set, res, average='binary')

print(p)
print(r)
print(f1score)

# (1270, 900)
length = 50
index_array = np.arange(length)
for j in range(100):
    data_array = motion[j*10].reshape(50, 18)
    print(np.shape(data_array))
    #mix two datagram: original data visualization
    fig, axs = plt.subplots(9, 1)
    for i in range(9):
        axs[i].plot(index_array, data_array[:,i], index_array, data_array[:,i+9])
    plt.show()
