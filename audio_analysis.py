import numpy as np
from scipy.io import wavfile
from scipy.fftpack import fft,ifft
import matplotlib.pylab as plt
from sensor_module import Process
from audio_module import AudioProcess
from python_speech_features  import mfcc
import random

#initialize
left_sensor = Process('data/sound/log-20190310-PxB-WatchL.txt')
left_sensor.read_data()
left_sensor.preprocess_timing_gap()
# left.show_single_plot()
right_sensor = Process('data/sound/log-20190310-PxB-WatchR.txt')
right_sensor.read_data()
right_sensor.preprocess_timing_gap()

FILE_SHIFT = 0
TIMING_DIFF = left_sensor.time[0] - right_sensor.time[0]
right_sensor.time = [time+TIMING_DIFF-FILE_SHIFT for time in right_sensor.time]

left_audio = AudioProcess('data/sound/log-20190310-PxB-WatchL.wav')
left_audio.frequency_transform()
left_audio.mfcc_transform()
right_audio = AudioProcess('data/sound/log-20190310-PxB-WatchR.wav')
right_audio.frequency_transform()
right_audio.mfcc_transform()

fig, axs = plt.subplots(4, 1)
axs[0].plot(range(len(left_sensor.time)), [data[0] for data in left_sensor.data['acc']], range(len(right_sensor.time)), [data[0] for data in right_sensor.data['acc']])
axs[1].plot(range(len(left_sensor.time)), [data[1] for data in left_sensor.data['acc']], range(len(right_sensor.time)), [data[1] for data in right_sensor.data['acc']])
axs[2].plot(range(len(left_sensor.time)), [data[2] for data in left_sensor.data['acc']], range(len(right_sensor.time)), [data[2] for data in right_sensor.data['acc']])
axs[3].plot(range(left_audio.audio.shape[0]), left_audio.audio, range(right_audio.audio.shape[0]), right_audio.audio)
plt.show()

#align
left_sensor_start = 465
right_sensor_start = 464
left_audio_start = 169690
right_audio_start = 172731

fig, axs = plt.subplots(4, 1)
axs[0].plot(range(len(left_sensor.time)-left_sensor_start), [data[0] for data in left_sensor.data['acc'][left_sensor_start:]], range(len(right_sensor.time)-right_sensor_start), [data[0] for data in right_sensor.data['acc'][right_sensor_start:]])
axs[1].plot(range(len(left_sensor.time)-left_sensor_start), [data[1] for data in left_sensor.data['acc'][left_sensor_start:]], range(len(right_sensor.time)-right_sensor_start), [data[1] for data in right_sensor.data['acc'][right_sensor_start:]])
axs[2].plot(range(len(left_sensor.time)-left_sensor_start), [data[2] for data in left_sensor.data['acc'][left_sensor_start:]], range(len(right_sensor.time)-right_sensor_start), [data[2] for data in right_sensor.data['acc'][right_sensor_start:]])
axs[3].plot(range(left_audio.audio.shape[0]-left_audio_start), left_audio.audio[left_audio_start:], range(right_audio.audio.shape[0]-right_audio_start), right_audio.audio[right_audio_start:])
plt.show()

#############display
#transform to frequency field by fft(sample n = 50 -> 50Hz, peek frequency = 20Hz, enough)
left_sensor.frequency_transform()
left_sensor.frequency_bucket_show()
right_sensor.frequency_transform()
right_sensor.frequency_bucket_show()
#display audio freq
# fig, axs = plt.subplots(3, 1)
# axs[0].plot(left_audio.freq_energy_array[:, 0])
# axs[1].plot(left_audio.freq_energy_array[:, 1])
# axs[2].plot(left_audio.freq_energy_array[:, 2])
# plt.show()

# fig, axs = plt.subplots(3, 1)
# axs[0].plot(right_audio.freq_energy_array[:, 0])
# axs[1].plot(right_audio.freq_energy_array[:, 1])
# axs[2].plot(right_audio.freq_energy_array[:, 2])
# plt.show()


#initialization
left_sensor_index = left_sensor_start
right_sensor_index = right_sensor_start
left_audio_index = left_audio_start
right_audio_index = right_audio_start

#detection
print('detectiondetectiondetectiondetection')
AUDIO_FREQ = 44100
SENSOR_FFT_THRESHOLD = 6#change
SENSOR_TIME_THRESHOLD = 0.6#change
AUDIO_FFT_THRESHOLD = 20#change
AUDIO_TIME_THRESHOLD = 0.10#change

length = 50
offset = 25
audio_length = 22050
cover_array = []
fft_cover_array = []
audio_cover_array = []
fft_audio_cover_array = []
count = 0
fft_count = 0
audio_count = 0
fft_audio_count = 0
all_count = 0
index_array = np.arange(0, 50)
store_data_list = []

def find_peak_index(data_unit):
    data_length = data_unit.shape[0]
    norm_unit = np.zeros((data_length, 2))
    for i in range(data_length):
        norm_unit[i, 0] = np.linalg.norm(data_unit[i, 0:3])
        norm_unit[i, 1] = np.linalg.norm(data_unit[i, 9:12])
    # print(norm_unit)
    return int (np.argmax(norm_unit) / 2) - 25

def mfcc_transform(segment, sampling_freq, fft_size):
    mfcc_array = mfcc(segment, samplerate=sampling_freq, winlen=0.5, winstep=0.5, nfft=fft_size)
    print(mfcc_array.shape)
    return mfcc_array

while left_sensor_index + length < len(left_sensor.time) and right_sensor_index + length < len(right_sensor.time):
    left_audio_index = int((left_sensor.time[left_sensor_index] - left_sensor.time[left_sensor_start]) * AUDIO_FREQ) + left_audio_start
    right_audio_index = int((right_sensor.time[right_sensor_index] - right_sensor.time[right_sensor_start]) * AUDIO_FREQ) + right_audio_start
    if left_audio_index + audio_length > left_audio.audio.shape[0] or right_audio_index + audio_length > right_audio.audio.shape[0]:
        print('out of bound')
        break
    
    left_sensor_segment = (np.array(left_sensor.data['acc'][left_sensor_index: left_sensor_index+length])).T
    right_sensor_segment = (np.array(right_sensor.data['acc'][right_sensor_index: right_sensor_index+length])).T
    left_audio_segment = left_audio.audio.data[left_audio_index : left_audio_index + audio_length]
    right_audio_segment = right_audio.audio.data[right_audio_index : right_audio_index + audio_length]

    #judge four facets
    #1. sensor time series
    # print(left_sensor_segment.shape, right_sensor_segment.shape)
    is_sensor_time_peak = False
    if np.max(np.fabs(left_sensor_segment)) > SENSOR_TIME_THRESHOLD or np.max(np.fabs(right_sensor_segment)) > SENSOR_TIME_THRESHOLD:#change
        is_sensor_time_peak = True
        cover_array.append(1)
        count = count + 1
    else:
        cover_array.append(0)
    
    #2. sensor freq series
    left_data_fft = np.zeros(left_sensor_segment.shape)
    right_data_fft = np.zeros(right_sensor_segment.shape)
    for i in range(3):
        left_data_fft[i] = np.array(abs(fft(left_sensor_segment[i])))
        right_data_fft[i] = np.array(abs(fft(right_sensor_segment[i])))
    left_bucket = [[0,0,0], [0,0,0], [0,0,0]]
    right_bucket = [[0,0,0], [0,0,0], [0,0,0]]
    for j in range(3):
        for k in range((int)(length/2)):
            left_bucket[j][(int)((k+5)/10)] = left_bucket[j][(int)((k+5)/10)] + (left_data_fft[j, k])
            right_bucket[j][(int)((k+5)/10)] = right_bucket[j][(int)((k+5)/10)] + (right_data_fft[j, k])
    is_sensor_freq_peak = False
    for j in range(3):
        if left_bucket[i][1] > SENSOR_FFT_THRESHOLD or right_bucket[i][1] > SENSOR_FFT_THRESHOLD:#change 现在取的是5-15hz的能量值判断。
            is_sensor_freq_peak = True
        break
    if is_sensor_freq_peak:
        fft_cover_array.append(1)
        fft_count = fft_count + 1
    else:
        fft_cover_array.append(0)
    
    #3. audio time series
    is_audio_time_peak = False
    if np.max(np.fabs(left_audio_segment)) > AUDIO_TIME_THRESHOLD or np.max(np.fabs(right_audio_segment)) > AUDIO_TIME_THRESHOLD:#change
        is_audio_time_peak = True
        audio_cover_array.append(1)
        audio_count = audio_count + 1
    else:
        audio_cover_array.append(0)

    #4. audio freq series
    # is_audio_freq_peak = False
    # left_audio_freq = np.array(abs(fft(left_audio_segment)))
    # right_audio_freq = np.array(abs(fft(right_audio_segment)))
    # print(left_audio_freq.shape, right_audio_freq.shape)
    # lbound = 100
    # ubound = 1000
    # if np.sum(left_audio_freq[lbound:ubound]) > AUDIO_FFT_THRESHOLD or np.sum(right_audio_freq[lbound:ubound]) > AUDIO_FFT_THRESHOLD:
    #     is_audio_freq_peak = True
    #     fft_audio_count = fft_audio_count + 1
    #     fft_audio_cover_array.append(1)
    # else:
    #     fft_audio_cover_array.append(0)
    
    #judge
    if is_sensor_time_peak and is_sensor_freq_peak and is_audio_time_peak:
        all_count = all_count + 1
        print('detect a gesture')
        #record
        store_data = []
        for i in range(length):
            for j in range(3):
                store_data.append(left_sensor.data['acc'][left_sensor_index+i][j])
            for j in range(3):
                store_data.append(left_sensor.data['att'][left_sensor_index+i][j])
            for j in range(3):
                store_data.append(left_sensor.data['rot'][left_sensor_index+i][j])
            for j in range(3):
                store_data.append(right_sensor.data['acc'][right_sensor_index+i][j])
            for j in range(3):
                store_data.append(right_sensor.data['att'][right_sensor_index+i][j])
            for j in range(3):
                store_data.append(right_sensor.data['rot'][right_sensor_index+i][j])
        data_unit = (np.array(store_data)).reshape(50, 18)
        peak_index = find_peak_index(data_unit) + int(random.random()*20-10)
        left_start = left_sensor_index + peak_index
        right_start = right_sensor_index + peak_index
        store_data = []
        for k in range(length):
            for j in range(3):
                store_data.append(left_sensor.data['acc'][left_start+k][j])
            for j in range(3):
                store_data.append(left_sensor.data['att'][left_start+k][j])
            for j in range(3):
                store_data.append(left_sensor.data['rot'][left_start+k][j])
            for j in range(3):
                store_data.append(right_sensor.data['acc'][right_start+k][j])
            for j in range(3):
                store_data.append(right_sensor.data['att'][right_start+k][j])
            for j in range(3):
                store_data.append(right_sensor.data['rot'][right_start+k][j])
        #left_sensor_segment, right_sensor_segment, 
        store_data = np.array(store_data).reshape(-1)
        # left_audio_freq = np.array(abs(fft(left_audio_segment)))
        # right_audio_freq = np.array(abs(fft(right_audio_segment)))
        # left_mfcc_array = mfcc_transform(left_audio_segment, left_audio.sampling_freq, left_audio.fft_size)
        # right_mfcc_array = mfcc_transform(right_audio_segment, right_audio.sampling_freq, right_audio.fft_size)
        store_data_list.append(store_data)

    left_sensor_index = left_sensor_index + length
    right_sensor_index = right_sensor_index + length

print(count, fft_count, audio_count, all_count)

fig, axs = plt.subplots(6, 1)
axs[0].plot(left_sensor.time, [data[0] for data in left_sensor.data['acc']], right_sensor.time, [data[0] for data in right_sensor.data['acc']])
axs[1].plot(left_sensor.time, [data[1] for data in left_sensor.data['acc']], right_sensor.time, [data[1] for data in right_sensor.data['acc']])
axs[2].plot(left_sensor.time, [data[2] for data in left_sensor.data['acc']], right_sensor.time, [data[2] for data in right_sensor.data['acc']])
axs[3].plot(cover_array)
axs[4].plot(fft_cover_array)
axs[5].plot(audio_cover_array)
plt.show()

