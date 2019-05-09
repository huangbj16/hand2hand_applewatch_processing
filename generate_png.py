import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal

file_path1 = 'D:/hand2hand_apple/training/sound_new/hbj/PxB_np.npy'
file_path2 = 'D:/hand2hand_apple/training/sound_new/hbj/IxB_np.npy'
dataset1 = np.load(file_path1)
dataset2 = np.load(file_path2)
print(dataset1.shape, dataset2.shape)

sensor_length = 1000
audio_length = 22050
fs = 44100
cm=plt.cm.get_cmap('gray')
axis = ['x', 'y', 'z']

for i in range(28, dataset1.shape[0]):
    # fig, axs = plt.subplots(3, 2)
    segment = dataset1[i]
    data_unit = segment[0 : sensor_length].reshape(50, 20)
    audio_left1 = segment[sensor_length : sensor_length+audio_length]
    audio_right1 = segment[sensor_length+audio_length : sensor_length+2*audio_length]
    plt.subplot(321)  
    plt.plot(range(50), data_unit[:, 0], range(50), data_unit[:, 10+0])
    plt.ylim((-1.5, 1.5))
    # plt.ylabel(axis[0])

    plt.subplot(323)  
    plt.plot(range(50), data_unit[:, 1], range(50), data_unit[:, 10+1])
    plt.ylim((-2, 2))
    # plt.ylabel(axis[1])

    plt.subplot(325)  
    plt.plot(range(50), data_unit[:, 2], range(50), data_unit[:, 10+2])
    plt.ylim((-3, 3))
    # plt.ylabel(axis[2])
    ###
    segment = dataset2[i]
    data_unit = segment[0 : sensor_length].reshape(50, 20)
    audio_left2 = segment[sensor_length : sensor_length+audio_length]
    audio_right2 = segment[sensor_length+audio_length : sensor_length+2*audio_length]

    plt.subplot(322)  
    plt.plot(range(50), data_unit[:, 0], range(50), data_unit[:, 10+0])
    plt.ylim((-1.5, 1.5))
    # plt.title('IxB Acceleration\nAxis '+axis[0])

    plt.subplot(324)  
    plt.plot(range(50), data_unit[:, 1], range(50), data_unit[:, 10+1])
    plt.ylim((-2, 2))
    # plt.title('Axis '+axis[1])

    plt.subplot(326)  
    plt.plot(range(50), data_unit[:, 2], range(50), data_unit[:, 10+2])
    # plt.title('Axis '+axis[2])
    plt.ylim((-3, 3))
    plt.subplots_adjust(hspace = 0.5)
    plt.show()




    #########feature: stft
    ##############display
    f, t, Zxx = signal.stft(audio_left1, fs, nperseg=420)
    Zxx = np.abs(Zxx)   
    print(f[:10])
    plt.subplot(121)  
    plt.pcolormesh(t, f[:10], np.abs(Zxx)[:10], vmin=-0.04, vmax=0.04, cmap = cm)
    plt.colorbar()
    # plt.title('PxB left')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    # f, t, Zxx = signal.stft(audio_right1, fs, nperseg=420)
    # Zxx = np.abs(Zxx)     
    # plt.subplot(427)
    # plt.pcolormesh(t, f[:10], np.abs(Zxx)[:10], vmin=-0.04, vmax=0.04, cmap = cm)
    # plt.colorbar()
    # plt.title('PxB right')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')

    # plt.show()
    ##############display
    f, t, Zxx = signal.stft(audio_left2, fs, nperseg=420)
    Zxx = np.abs(Zxx) 
    print(f[:10])  
    plt.subplot(122)  
    plt.pcolormesh(t, f[:10], np.abs(Zxx)[:10], vmin=-0.04, vmax=0.04, cmap = cm)
    plt.colorbar()
    # plt.title('IxB left')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    # f, t, Zxx = signal.stft(audio_right2, fs, nperseg=420)
    # Zxx = np.abs(Zxx)     
    # plt.subplot(428)
    # plt.pcolormesh(t, f[:10], np.abs(Zxx)[:10], vmin=-0.04, vmax=0.04, cmap = cm)
    # plt.colorbar()
    # plt.title('IxB right')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')

    plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
    plt.show()
    # exit(0)
    