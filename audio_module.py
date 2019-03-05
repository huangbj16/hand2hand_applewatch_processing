import matplotlib.pyplot as plt
import math as math
import numpy as np
import random
from scipy.fftpack import fft,ifft
from scipy.io import wavfile

class AudioProcess(object):

    def __init__(self, filename):
        self.filename = filename
        sampling_freq, audio = wavfile.read(filename)
        print('\nShape:', audio.shape)
        print('Datatype:', audio.dtype)
        print('Duration:', round(audio.shape[0] / float(sampling_freq), 3), 'seconds')
        print('sampling: ', sampling_freq)
        self.audio = audio / (2.**15)
        self.sampling_freq = sampling_freq
        plt.plot(self.audio)
        plt.show()

    def frequency_transform(self):
        fft_size = 32768
        round = int(self.audio.shape[0] / fft_size)
        self.round = round
        self.freq_array = np.zeros((self.round, fft_size))
        for i in range(round-1):
            audio_segment = self.audio[i*fft_size : (i+1)*fft_size]
            frequ_segment = abs(fft(audio_segment))
            self.freq_array[i] = frequ_segment
    
        