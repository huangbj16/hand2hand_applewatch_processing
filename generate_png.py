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
    fig, axs = plt.subplots(3, 2)
    segment = dataset1[i]
    data_unit = segment[0 : sensor_length].reshape(50, 20)
    audio_left1 = segment[sensor_length : sensor_length+audio_length]
    audio_right1 = segment[sensor_length+audio_length : sensor_length+2*audio_length]
    for j in range(3):
        axs[j, 0].plot(range(50), data_unit[:, j], range(50), data_unit[:, 10+j])
    axs[0, 0].set_title('PxB Acceleration\nAxis '+axis[0])
    axs[1, 0].set_title('Axis '+axis[1])
    axs[2, 0].set_title('Axis '+axis[2])
    ###
    segment = dataset2[i]
    data_unit = segment[0 : sensor_length].reshape(50, 20)
    audio_left2 = segment[sensor_length : sensor_length+audio_length]
    audio_right2 = segment[sensor_length+audio_length : sensor_length+2*audio_length]
    for j in range(3):
        axs[j, 1].plot(range(50), data_unit[:, j], range(50), data_unit[:, 10+j])
    axs[0, 1].set_title('IxB Acceleration\nAxis '+axis[0])
    axs[1, 1].set_title('Axis '+axis[1])
    axs[2, 1].set_title('Axis '+axis[2])
    # plt.setp(axs, ylim=(-5, 5))
    plt.subplots_adjust(hspace = 0.5)
    plt.show()




    #########feature: stft
    ##############display
    f, t, Zxx = signal.stft(audio_left1, fs, nperseg=420)
    Zxx = np.abs(Zxx)   
    print(f[:10])
    plt.subplot(221)  
    plt.pcolormesh(t, f[:10], np.abs(Zxx)[:10], vmin=-0.04, vmax=0.04, cmap = cm)
    plt.colorbar()
    plt.title('PxB left')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    f, t, Zxx = signal.stft(audio_right1, fs, nperseg=420)
    Zxx = np.abs(Zxx)     
    plt.subplot(223)
    plt.pcolormesh(t, f[:10], np.abs(Zxx)[:10], vmin=-0.04, vmax=0.04, cmap = cm)
    plt.colorbar()
    plt.title('PxB right')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    # plt.show()
    ##############display
    f, t, Zxx = signal.stft(audio_left2, fs, nperseg=420)
    Zxx = np.abs(Zxx) 
    print(f[:10])  
    plt.subplot(222)  
    plt.pcolormesh(t, f[:10], np.abs(Zxx)[:10], vmin=-0.04, vmax=0.04, cmap = cm)
    plt.colorbar()
    plt.title('IxB left')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    f, t, Zxx = signal.stft(audio_right2, fs, nperseg=420)
    Zxx = np.abs(Zxx)     
    plt.subplot(224)
    plt.pcolormesh(t, f[:10], np.abs(Zxx)[:10], vmin=-0.04, vmax=0.04, cmap = cm)
    plt.colorbar()
    plt.title('IxB right')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
    plt.show()
    # exit(0)
    