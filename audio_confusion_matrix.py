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
type_num = len(type_array)

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

#####################feature process
feature_array = []

for i in range(len(type_array)):
    primitive_data = type_array[i]
    data_length = primitive_data.shape[0]
    bound = 20
    feature_length = 20
    featured_data = np.zeros((data_length, feature_length))
    print('type:', motion_type[i])
    for j in range(data_length):
        segment = primitive_data[j]
        data_unit = segment[0:900].reshape(50, 18)
        audio_left = segment[900:900+22050]
        audio_right = segment[900+22050:900+44100]
        freq_audio_left = np.array(abs(fft(audio_left)))
        freq_audio_right = np.array(abs(fft(audio_right)))

        
        # freq_audio_left = freq_audio_left[:5000]
        # freq_audio_right = freq_audio_right[:5000]

        ##############feature: stft
        ##############display
        # if i != 6:
        #     break
        # fs = 44100
        # f, t, Zxx = signal.stft(audio_left, fs, nperseg=100)
        # plt.pcolormesh(t, f[:10], np.abs(Zxx)[:10], vmin=np.min(audio_left), vmax=np.max(audio_left))
        # plt.title('STFT Magnitude')
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        # plt.show()
        # continue

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


        ##############feature: bucket
        ######normalize
        freq_audio_left = freq_audio_left / np.linalg.norm(freq_audio_left)
        freq_audio_right = freq_audio_right / np.linalg.norm(freq_audio_right)
        bucket_index = [0, 100, 200, 400, 800, 1200, 1600, 2400, 3200, 6400, 11025]
        # bucket_index = [0, 50, 100, 150, 200, 400, 600, 800, 1000, 2000, 11025]
        for k in range(len(bucket_index)-1):
            lower_bound = bucket_index[k]
            upper_bound = bucket_index[k+1]
            featured_data[j, k] = np.sum(freq_audio_left[lower_bound:upper_bound])
            featured_data[j, 10+k] = np.sum(freq_audio_right[lower_bound:upper_bound])
        ##############feature: uniform bucket
        # freq_energy_audio_left = np.sum(freq_audio_left, axis=1)
        # freq_energy_audio_right = np.sum(freq_audio_right, axis=1)
        # print(freq_energy_audio_left.shape, freq_energy_audio_right.shape)
        # featured_data[j, 0:bound] = freq_energy_audio_left[0:bound]
        # featured_data[j, bound:2*bound] = freq_energy_audio_right[0:bound]
        

        ##############feature: mfcc
        # sampling_freq = 44100
        # fft_size = 22050
        # mfcc_left = mfcc(audio_left, samplerate=sampling_freq, winlen=0.25, winstep=0.125, nfft=fft_size)
        # mfcc_right = mfcc(audio_right, samplerate=sampling_freq, winlen=0.25, winstep=0.125, nfft=fft_size)
        # # print(mfcc_left.shape, mfcc_right.shape)
        # featured_data[j, 0:bound] = mfcc_left.reshape(-1)
        # featured_data[j, bound:2*bound] = mfcc_right.reshape(-1)


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

res_matrix = np.zeros((type_num, type_num), dtype=np.int)

for i in range(100):
    clf = SVC(kernel='rbf', gamma='auto')# ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
    # print('current time: ', time.time())
    seed = int(time.time()*10000000) % 19980608
    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(feature_set, flag_set, test_size = 0.2, random_state=seed)
    # Run classifier, using a model that is too regularized (C too low) to see
    # the impact on the results
    y_pred = clf.fit(X_train, y_train).predict(X_test)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    res_matrix = res_matrix + cnf_matrix
    np.set_printoptions(precision=2)

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


class_names = np.arange(type_num)#change

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(res_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix')

plt.show()
