import numpy as np
from scipy.io import wavfile
from scipy.fftpack import fft,ifft
import matplotlib.pylab as plt
from sensor_module import Process
from audio_module import AudioProcess

#initialize
left_sensor = Process('data/sound/log-run&IxB-WatchL.txt')
left_sensor.read_data()
left_sensor.preprocess_timing_gap()
# left.show_single_plot()
right_sensor = Process('data/sound/log-run&IxB-WatchR.txt')
right_sensor.read_data()
right_sensor.preprocess_timing_gap()

left_audio = AudioProcess('data/sound/log-run&IxB-WatchL.wav')
left_audio.frequency_transform()
right_audio = AudioProcess('data/sound/log-run&IxB-WatchR.wav')
right_audio.frequency_transform()

fig, axs = plt.subplots(3, 1)

