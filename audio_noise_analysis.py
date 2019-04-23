import numpy as np
from scipy.io import wavfile
from scipy.fftpack import fft,ifft
import matplotlib.pylab as plt
from sensor_module import Process
from audio_module import AudioProcess
import random
from lyq_quaternion_qua import rotate2

is_display_on = True
is_single_display_on = False
ready_for_save = True

#initialize
left_sensor = Process('data/sound_final/swn/log-20190422-170557-WatchL.txt')
left_sensor.read_data()
left_sensor.preprocess_timing_gap()
# left_sensor.show_single_plot()
right_sensor = Process('data/sound_final/swn/log-20190422-170557-WatchR.txt')
right_sensor.read_data()
right_sensor.preprocess_timing_gap()
# right_sensor.show_single_plot()

TIMING_DIFF = left_sensor.time[0] - right_sensor.time[0]
right_sensor.time = [time+TIMING_DIFF for time in right_sensor.time]

left_audio = AudioProcess('data/sound_final/swn/log-20190422-170557-WatchL.wav')
left_audio.frequency_transform()
left_audio.mfcc_transform()
right_audio = AudioProcess('data/sound_final/swn/log-20190422-170557-WatchR.wav')
right_audio.frequency_transform()
right_audio.mfcc_transform()

if is_display_on:
    fig, axs = plt.subplots(4, 1)
    axs[0].plot(range(len(left_sensor.time)), [data[0] for data in left_sensor.data['acc']], range(len(right_sensor.time)), [data[0] for data in right_sensor.data['acc']])
    axs[1].plot(range(len(left_sensor.time)), [data[1] for data in left_sensor.data['acc']], range(len(right_sensor.time)), [data[1] for data in right_sensor.data['acc']])
    axs[2].plot(range(len(left_sensor.time)), [data[2] for data in left_sensor.data['acc']], range(len(right_sensor.time)), [data[2] for data in right_sensor.data['acc']])
    axs[3].plot(range(left_audio.audio.shape[0]), left_audio.audio, range(right_audio.audio.shape[0]), right_audio.audio)
    plt.show()

#align
left_sensor_start = 0
right_sensor_start = 0
left_audio_start = 0
right_audio_start = 0
is_autoalign = True#change
#auto align
if is_autoalign:
    autoalign_threshold_sensor = 4#change
    autoalign_threshold_audio = 0.2#change
    for unit_index in range(len(left_sensor.time)):
        unit = left_sensor.data['acc'][unit_index]
        if np.max(np.fabs(unit)) > autoalign_threshold_sensor:
            #pick the peak
            segment = left_sensor.data['acc'][unit_index-25: unit_index+25]
            segment_index = np.argmax(np.fabs(segment))
            # print(segment)
            # print(unit_index, segment_index)
            left_sensor_start = unit_index + int(segment_index / 3) + 50
            break
    for unit_index in range(len(right_sensor.time)):
        unit = right_sensor.data['acc'][unit_index]
        if np.max(np.fabs(unit)) > autoalign_threshold_sensor:
            #pick the peak
            segment = right_sensor.data['acc'][unit_index-25: unit_index+25]
            segment_index = np.argmax(np.fabs(segment)) 
            # print(segment)
            # print(unit_index, segment_index)
            right_sensor_start = unit_index + int(segment_index / 3) + 50
            break
    #audio offset to ignore the initial noise.
    audio_initial_offset = 11025
    for unit_index in range(len(left_audio.audio)-audio_initial_offset):
        unit = left_audio.audio[unit_index+audio_initial_offset]
        if np.max(np.fabs(unit)) > autoalign_threshold_audio:
            #pick the peak
            segment = left_audio.audio[unit_index-11025: unit_index+11025]
            segment_index = np.argmax(np.fabs(segment)) % 22050
            left_audio_start = unit_index + segment_index + 22050
            break
    for unit_index in range(len(right_audio.audio)-audio_initial_offset):
        unit = right_audio.audio[unit_index+audio_initial_offset]
        if np.max(np.fabs(unit)) > autoalign_threshold_audio:
            #pick the peak
            segment = right_audio.audio[unit_index-11025: unit_index+11025]
            segment_index = np.argmax(np.fabs(segment)) % 22050
            right_audio_start = unit_index + segment_index + 22050
            break
else:
    left_sensor_start = 0
    right_sensor_start = 0
    left_audio_start = 0
    right_audio_start = 0

print('autoalign result: ', left_sensor_start, right_sensor_start, left_audio_start, right_audio_start)
print('start time: ', left_sensor.time[left_sensor_start], right_sensor.time[right_sensor_start])
TIMING_DIFF = left_sensor.time[left_sensor_start] - right_sensor.time[right_sensor_start]
right_sensor.time = [time+TIMING_DIFF for time in right_sensor.time]
print('start time: ', left_sensor.time[left_sensor_start], right_sensor.time[right_sensor_start])

def calculate_correlation(acc_left, acc_right, qua_left, qua_right):
    calculate_length = min(len(acc_left), len(acc_right))
    acc_left = np.array(acc_left[:calculate_length])
    acc_right = np.array(acc_right[:calculate_length])
    qua_left = np.array(qua_left[:calculate_length]) * (-1)
    qua_right = np.array(qua_right[:calculate_length]) * (-1)
    acc_left_rotated = rotate2(acc_left, qua_left, qua_right)
    print(acc_left_rotated.shape, acc_right.shape)
    correlation = acc_left_rotated * acc_right * (-1)
    print(correlation.shape)
    return correlation, acc_left_rotated, acc_right, calculate_length
    
acc_correlation, acc_left_rotated, acc_right, calculate_length = calculate_correlation(left_sensor.data['acc'][left_sensor_start:], right_sensor.data['acc'][right_sensor_start:], left_sensor.data['qua'][left_sensor_start:], right_sensor.data['qua'][right_sensor_start:])

#############display raw data and correlation
if is_display_on:
    fig, axs = plt.subplots(4, 1)
    # plt.setp(axs, ylim=(-3, 3))
    # axs[0].plot(left_sensor.time[left_sensor_start:], [data[0] for data in left_sensor.data['acc'][left_sensor_start:]], right_sensor.time[right_sensor_start:], [data[0] for data in right_sensor.data['acc'][right_sensor_start:]])
    # axs[1].plot(acc_correlation[:, 0])
    # axs[2].plot(left_sensor.time[left_sensor_start:], [data[1] for data in left_sensor.data['acc'][left_sensor_start:]], right_sensor.time[right_sensor_start:], [data[1] for data in right_sensor.data['acc'][right_sensor_start:]])
    # axs[3].plot(acc_correlation[:, 1])
    # axs[4].plot(left_sensor.time[left_sensor_start:], [data[2] for data in left_sensor.data['acc'][left_sensor_start:]], right_sensor.time[right_sensor_start:], [data[2] for data in right_sensor.data['acc'][right_sensor_start:]])
    # axs[5].plot(acc_correlation[:, 2])
    axs[0].plot(range(calculate_length), acc_left_rotated[:, 0], range(calculate_length), acc_right[:, 0])
    # axs[1].plot(acc_correlation[:, 0])
    axs[1].plot(range(calculate_length), acc_left_rotated[:, 1], range(calculate_length), acc_right[:, 1])
    # axs[3].plot(acc_correlation[:, 1])
    axs[2].plot(range(calculate_length), acc_left_rotated[:, 2], range(calculate_length), acc_right[:, 2])
    # axs[5].plot(acc_correlation[:, 2])
    axs[3].plot(range(left_audio.audio.shape[0]-left_audio_start), left_audio.audio[left_audio_start:], range(right_audio.audio.shape[0]-right_audio_start), right_audio.audio[right_audio_start:])
    plt.show()
    

#############display
#transform to frequency field by fft(sample n = 50 -> 50Hz, peek frequency = 20Hz, enough)
left_sensor.frequency_transform()
if is_display_on:
    left_sensor.frequency_bucket_show()
right_sensor.frequency_transform()
if is_display_on:
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

length = 50
offset = 25
audio_length = 22050
index_array = np.arange(0, 50)
store_data_list = []
gesture_count = 0

left_sensor_random_range = len(left_sensor.time) - left_sensor_start - 100
right_sensor_random_range = len(right_sensor.time) - right_sensor_start - 100
sensor_random_range = min(left_sensor_random_range, right_sensor_random_range)

print('random range: ', sensor_random_range)

while gesture_count < 50:
    offset = random.randint(0, sensor_random_range)
    left_sensor_index = left_sensor_start + offset
    right_sensor_index = right_sensor_start + offset
    while abs(left_sensor.time[left_sensor_index] - right_sensor.time[right_sensor_index]) > 0.01:
        if left_sensor.time[left_sensor_index] < right_sensor.time[right_sensor_index]:
            left_sensor_index = left_sensor_index + 1
        else:
            right_sensor_index = right_sensor_index + 1
    left_audio_index = int((left_sensor.time[left_sensor_index] - left_sensor.time[left_sensor_start]) * AUDIO_FREQ) + left_audio_start
    right_audio_index = int((right_sensor.time[right_sensor_index] - right_sensor.time[right_sensor_start]) * AUDIO_FREQ) + right_audio_start
    # print(left_sensor_index, right_sensor_index, left_audio_index, right_audio_index)
    if left_audio_index + audio_length > left_audio.audio.shape[0] or right_audio_index + audio_length > right_audio.audio.shape[0] or left_sensor_index + length > len(left_sensor.time) or right_sensor_index + length > len(right_sensor.time):
        print('out of bound')
        continue
    
    left_audio_segment = left_audio.audio.data[left_audio_index : left_audio_index + audio_length]
    right_audio_segment = right_audio.audio.data[right_audio_index : right_audio_index + audio_length]

    #record
    store_data = np.zeros((20, 50))
    store_data[0:3] = (np.array(left_sensor.data['acc'][left_sensor_index: left_sensor_index+length]).T)
    store_data[3:6] = (np.array(left_sensor.data['rot'][left_sensor_index: left_sensor_index+length]).T)
    store_data[6:10] = (np.array(left_sensor.data['qua'][left_sensor_index: left_sensor_index+length]).T)
    store_data[10:13] = (np.array(right_sensor.data['acc'][right_sensor_index: right_sensor_index+length]).T)
    store_data[13:16] = (np.array(right_sensor.data['rot'][right_sensor_index: right_sensor_index+length]).T)
    store_data[16:20] = (np.array(right_sensor.data['qua'][right_sensor_index: right_sensor_index+length]).T)
    data_unit = store_data.T
    gesture_count = gesture_count + 1
    store_data = (store_data.T).reshape(-1)
    store_data = np.concatenate((store_data, left_audio_segment, right_audio_segment))
    store_data_list.append(store_data)
    
    # display
    if is_single_display_on:
        fig, axs = plt.subplots(4, 2)
        for i in range(3):
            axs[i][0].plot(store_data[0:3])
            axs[i][1].plot(store_data[10:13])
        axs[3][0].plot(left_audio_segment)
        axs[3][1].plot(right_audio_segment)
        plt.show()

print(len(store_data_list))
np.save('training/sound_final/swn/combination/noise_np', store_data_list)

