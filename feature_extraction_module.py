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
from scipy.stats import entropy
from scipy import signal
import sys
sys.path.append('/')
from python_audio_feature import mfcc
from lyq_quaternion_qua import delta_qua

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

def feature_extraction_old(segment):
    feature_length = 72
    featured_unit = np.zeros((feature_length))

    data_unit = segment[0:900].reshape(50, 18)
    audio_left = segment[900:900+22050]
    audio_right = segment[900+22050:900+44100]
    freq_audio_left = np.array(abs(fft(audio_left)))
    freq_audio_right = np.array(abs(fft(audio_right)))

    feature_offset = 0
    for k in range(18):
        # if not isRot(k):
        data_unit_coor = data_unit[:, k]
        featured_unit[feature_offset + 4*k] = (int(np.min(data_unit_coor) * 1000)) / 1000
        featured_unit[feature_offset + 4*k+1] = (int(np.max(data_unit_coor) * 1000)) / 1000
        featured_unit[feature_offset + 4*k+2] = (int(np.mean(data_unit_coor) * 1000)) / 1000
        featured_unit[feature_offset + 4*k+3] = (int(np.std(data_unit_coor) * 1000)) / 1000

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
    # featured_unit = np.concatenate((audio_max, audio_min, audio_mean, audio_std))


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
    # featured_unit = np.concatenate((root_mean_square))

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
    # featured_unit[0:bound] = Zxx[:25]
    # f, t, Zxx = signal.stft(audio_right, fs, nperseg=2100)
    # Zxx = np.abs(Zxx)
    # Zxx = np.sum(Zxx, axis=1)
    # Zxx = Zxx / np.linalg.norm(Zxx)
    # featured_unit[bound:] = Zxx[:25]
    

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
    # featured_unit[:10] = peaks_left[:10]
    # featured_unit[10:20] = peaks_right[:10]


    ##############feature: bucket max min mean = : 0.9057750759878419 0.8206686930091185 0.8540425531914891
    ######normalize 
    # freq_audio_left = freq_audio_left / np.linalg.norm(freq_audio_left)
    # freq_audio_right = freq_audio_right / np.linalg.norm(freq_audio_right)
    # bucket_index = [0, 100, 200, 400, 800, 1600, 2400, 3200, 4000, 6400, 11025]
    # for k in range(len(bucket_index)-1):
    #     lower_bound = bucket_index[k]
    #     upper_bound = bucket_index[k+1]
    #     featured_unit[k] = np.sum(freq_audio_left[lower_bound:upper_bound])
    #     featured_unit[10+k] = np.sum(freq_audio_right[lower_bound:upper_bound])

    ##############feature: uniform bucket
    # freq_energy_audio_left = np.sum(freq_audio_left, axis=1)
    # freq_energy_audio_right = np.sum(freq_audio_right, axis=1)
    # print(freq_energy_audio_left.shape, freq_energy_audio_right.shape)
    # featured_unit[0:bound] = freq_energy_audio_left[0:bound]
    # featured_unit[bound:2*bound] = freq_energy_audio_right[0:bound]
    

    ##############feature: mfcc max min mean = : 0.9513677811550152 0.8844984802431611 0.9130699088145896
    # sampling_freq = 44100
    # fft_size = 22050
    # audio_left = audio_left / np.linalg.norm(audio_left)
    # audio_right = audio_right / np.linalg.norm(audio_right)
    # mfcc_left = mfcc(audio_left, samplerate=sampling_freq, winlen=0.5, winstep=0.25, nfft=fft_size)
    # mfcc_right = mfcc(audio_right, samplerate=sampling_freq, winlen=0.5, winstep=0.25, nfft=fft_size)
    # # print(mfcc_left.shape, mfcc_right.shape)
    # # print(np.mean(mfcc_left, axis=0).shape)
    # # exit(0)
    # featured_unit[0:bound] = np.abs(mfcc_left).reshape(-1)
    # featured_unit[bound:2*bound] = np.abs(mfcc_right).reshape(-1)
    # # featured_unit[0:bound] = np.amax(mfcc_left, axis=0)
    # # featured_unit[bound:2*bound] = np.amax(mfcc_right, axis=0)

    ###############feature: brute force audio
    # featured_unit = segment[900:]
    
    
    ##############feature: audio time (min, max, mean, std): accuracy 0.95
    # featured_unit[0] = np.min(audio_left)
    # featured_unit[1] = np.max(audio_left)
    # featured_unit[2] = np.mean(audio_left)
    # featured_unit[3] = np.std(audio_left)
    # featured_unit[4] = np.min(audio_right)
    # featured_unit[5] = np.max(audio_right)
    # featured_unit[6] = np.mean(audio_right)
    # featured_unit[7] = np.std(audio_right)

    return featured_unit

def feature_extraction_new(segment):
    bound = 26
    feature_length = 80+36
    featured_unit = np.zeros((feature_length))

    sensor_length = 1000
    audio_length = 22050
    data_unit = segment[0 : sensor_length].reshape(50, 20)
    audio_left = segment[sensor_length : sensor_length+audio_length]
    audio_right = segment[sensor_length+audio_length : sensor_length+2*audio_length]
    # freq_audio_left = np.array(abs(fft(audio_left)))
    # freq_audio_right = np.array(abs(fft(audio_right)))

    ##############display raw imu data
    # fig, axs = plt.subplots(10, 2)
    # for k in range(20):
    #     axs[k%10][int(k/10)].plot(data_unit[:, k])
    # # plt.setp(axs, ylim=(-20, 20))
    # plt.show()

    ###############calc delta_qua
    # qua_left = data_unit[:, 6:10]
    # qua_right = data_unit[:, 16:20]
    # sub_qua = np.zeros((50, 4), dtype=float)
    # for i in range(50):
    #     sub_qua[i] = delta_qua(qua_left[i], qua_right[i])
    # data_unit[:, 6:10] = sub_qua

    feature_offset = 0
    for k in range(20):
        data_unit_coor = data_unit[:, k]
        featured_unit[feature_offset + 4*k] = np.min(data_unit_coor)
        featured_unit[feature_offset + 4*k+1] = np.max(data_unit_coor)
        featured_unit[feature_offset + 4*k+2] = np.mean(data_unit_coor)
        featured_unit[feature_offset + 4*k+3] = np.std(data_unit_coor)
    
    ##########calc freq domain feature
    #acc freq
    acc_left = data_unit[:, 0:3] / np.linalg.norm(data_unit[:, 0:3])
    acc_right = data_unit[:, 10:13] / np.linalg.norm(data_unit[:, 10:13])
    acc_freq = np.concatenate((abs(fft(acc_left, axis=0)), abs(fft(acc_right, axis=0))), axis=1)
    freq_feature_offset = 80
    for k in range(6):
        acc_freq_coor = acc_freq[:, k]
        featured_unit[freq_feature_offset + k+6*0] = np.max(acc_freq_coor)
        featured_unit[freq_feature_offset + k+6*1] = np.argmax(acc_freq_coor)
        featured_unit[freq_feature_offset + k+6*2] = np.median(acc_freq_coor)
        featured_unit[freq_feature_offset + k+6*3] = np.mean(acc_freq_coor)
        featured_unit[freq_feature_offset + k+6*4] = np.std(acc_freq_coor)
        featured_unit[freq_feature_offset + k+6*5] = entropy(acc_freq_coor)

    # acc_left = data_unit[:, 0:3] / np.linalg.norm(data_unit[:, 0:3])
    # acc_right = data_unit[:, 10:13] / np.linalg.norm(data_unit[:, 10:13])
    # acc_comb = np.concatenate((acc_left, acc_right), axis=1)
    # freq_feature_offset = 80
    # sampling_freq = 100
    # fft_size = 50
    # for k in range(6):
    #     acc_coor = acc_comb[:, k]
    #     mfcc_acc_coor = mfcc(acc_coor, samplerate=sampling_freq, winlen=0.5, winstep=0.25, nfft=fft_size)
    #     featured_unit[freq_feature_offset + k*26: freq_feature_offset + (k+1)*26] = mfcc_acc_coor

    # feature_offset_peak = 0+80
    # for k in range(20):
    #     if k < 3:
    #         data_unit_coor = data_unit[:, k]
    #         window = 0
    #         peaks, _ = find_peaks(np.fabs(data_unit_coor), height=0.5)
    #         if len(peaks) == 0:
    #             window = -10
    #         else:
    #             window = (peaks[-1]-peaks[0])/len(peaks)
    #         featured_unit[feature_offset_peak + k] = window
    #     elif k >= 10 and k < 13:
    #         data_unit_coor = data_unit[:, k]
    #         window = 0
    #         peaks, _ = find_peaks(np.fabs(data_unit_coor), height=0.5)
    #         if len(peaks) == 0:
    #             window = -10
    #         else:
    #             window = (peaks[-1]-peaks[0])/len(peaks)
    #         featured_unit[feature_offset_peak + k-7] = window

    
    # ###############normalize acc & rot
    # index_start = [0, 3, 10, 13]
    # for i in index_start:
    #     sub_data_unit = data_unit[:, i:i+3]
    #     sub_max = np.max(np.fabs(sub_data_unit))
    #     data_unit[:, i:i+3] = sub_data_unit / sub_max

    # feature_offset_normalization = 52+80
    # for k in range(20):
    #     featured_unit[feature_offset_normalization + 4*k] = np.min(data_unit_coor)
    #     featured_unit[feature_offset_normalization + 4*k+1] = np.max(data_unit_coor)
    #     featured_unit[feature_offset_normalization + 4*k+2] = np.mean(data_unit_coor)
    #     featured_unit[feature_offset_normalization + 4*k+3] = np.std(data_unit_coor)

    ##############feature: mfcc max min mean = : 0.9513677811550152 0.8844984802431611 0.9130699088145896
    # sampling_freq = 44100
    # fft_size = 22050
    # audio_left = audio_left / np.linalg.norm(audio_left)
    # audio_right = audio_right / np.linalg.norm(audio_right)
    # mfcc_left = mfcc(audio_left, samplerate=sampling_freq, winlen=0.5, winstep=0.25, nfft=fft_size)
    # mfcc_right = mfcc(audio_right, samplerate=sampling_freq, winlen=0.5, winstep=0.25, nfft=fft_size)
    # # print(mfcc_left.shape, mfcc_right.shape)
    # # print(np.mean(mfcc_left, axis=0).shape)
    # # exit(0)
    # featured_unit[0:bound] = np.abs(mfcc_left).reshape(-1)
    # featured_unit[bound:2*bound] = np.abs(mfcc_right).reshape(-1)
    # # featured_unit[0:bound] = np.amax(mfcc_left, axis=0)
    # # featured_unit[bound:2*bound] = np.amax(mfcc_right, axis=0)

    return featured_unit

'''
| mfcc energy left | mfcc energy right | acc x left | acc y left |
| 0-------------25 | 26-------------51 | 52------55 | 56------59 |
'''