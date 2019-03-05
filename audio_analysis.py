import numpy as np
from scipy.io import wavfile
from scipy.fftpack import fft,ifft
import matplotlib.pylab as plt
from sensor_module import Process
from audio_module import AudioProcess

#initialize
left_sensor = Process('data/sound/log-PxP&IxP-WatchL.txt')
left_sensor.read_data()
left_sensor.preprocess_timing_gap()
# left.show_single_plot()
right_sensor = Process('data/sound/log-PxP&IxP-WatchR.txt')
right_sensor.read_data()
right_sensor.preprocess_timing_gap()

FILE_SHIFT = 0
TIMING_DIFF = left_sensor.time[0] - right_sensor.time[0]
right_sensor.time = [time+TIMING_DIFF-FILE_SHIFT for time in right_sensor.time]

left_audio = AudioProcess('data/sound/log-PxP&IxP-WatchL.wav')
left_audio.frequency_transform()
left_audio.mfcc_transform()
right_audio = AudioProcess('data/sound/log-PxP&IxP-WatchR.wav')
right_audio.frequency_transform()
right_audio.mfcc_transform()

fig, axs = plt.subplots(4, 1)
axs[0].plot(range(len(left_sensor.time)), [data[0] for data in left_sensor.data['acc']], range(len(right_sensor.time)), [data[0] for data in right_sensor.data['acc']])
axs[1].plot(range(len(left_sensor.time)), [data[1] for data in left_sensor.data['acc']], range(len(right_sensor.time)), [data[1] for data in right_sensor.data['acc']])
axs[2].plot(range(len(left_sensor.time)), [data[2] for data in left_sensor.data['acc']], range(len(right_sensor.time)), [data[2] for data in right_sensor.data['acc']])
axs[3].plot(range(left_audio.audio.shape[0]), left_audio.audio, range(right_audio.audio.shape[0]), right_audio.audio)
plt.show()

left_sensor_start = 606
right_sensor_start = 605
left_audio_start = 231867
right_audio_start = 159910

fig, axs = plt.subplots(4, 1)
axs[0].plot(range(len(left_sensor.time)-left_sensor_start), [data[0] for data in left_sensor.data['acc'][left_sensor_start:]], range(len(right_sensor.time)-right_sensor_start), [data[0] for data in right_sensor.data['acc'][right_sensor_start:]])
axs[1].plot(range(len(left_sensor.time)-left_sensor_start), [data[1] for data in left_sensor.data['acc'][left_sensor_start:]], range(len(right_sensor.time)-right_sensor_start), [data[1] for data in right_sensor.data['acc'][right_sensor_start:]])
axs[2].plot(range(len(left_sensor.time)-left_sensor_start), [data[2] for data in left_sensor.data['acc'][left_sensor_start:]], range(len(right_sensor.time)-right_sensor_start), [data[2] for data in right_sensor.data['acc'][right_sensor_start:]])
axs[3].plot(range(left_audio.audio.shape[0]-left_audio_start), left_audio.audio[left_audio_start:], range(right_audio.audio.shape[0]-right_audio_start), right_audio.audio[right_audio_start:])
plt.show()
