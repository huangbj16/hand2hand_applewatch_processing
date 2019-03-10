import matplotlib.pyplot as plt
import math as math
import numpy as np
import random
from scipy.fftpack import fft,ifft
from scipy.io import wavfile
from python_speech_features  import mfcc

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
        fft_size = 22050
        self.fft_size = fft_size
        round = int(self.audio.shape[0] / fft_size) - 1
        self.round = round
        self.freq_array = np.zeros((self.round, int(fft_size/2)))
        self.freq_energy_array = np.zeros((self.round, 3))
        for i in range(round):
            audio_segment = self.audio[i*fft_size : (i+1)*fft_size]
            frequ_segment = abs(fft(audio_segment))
            self.freq_array[i] = frequ_segment[: int(fft_size/2)]
            self.freq_energy_array[i] = [ np.sum(frequ_segment[0:100]), np.sum(frequ_segment[100:1000]), np.sum(frequ_segment[1000:2000]) ]
        print(self.freq_energy_array.shape)

    def mfcc_transform(self):
        self.mfcc_array = mfcc(self.audio, samplerate=self.sampling_freq, winlen=0.25, winstep=0.125, nfft=self.fft_size)
        print(self.mfcc_array.shape)



    
        