import matplotlib.pyplot as plt
import math as math
import numpy as np
import random
from scipy.fftpack import fft,ifft

def length(coor):
    x = float(coor[0])
    y = float(coor[1])
    z = float(coor[2])
    return math.sqrt(x*x+y*y+z*z)

class Process(object):

    def __init__(self, filename):
        self.filename = filename
        # self.array_time = []
        self.time = []
        self.data = {'acc':[], 'rot':[], 'qua':[]}

    def read_data(self):
        f = open(self.filename, 'r')
        lines = f.readlines()
        f.close()
        lineslen = int(len(lines)/4)
        print('%s  length = %d' % (self.filename, len(lines)))
        for line in range(lineslen):#4 lines each data
            self.time.append(float(lines[line*4].split()[1]))
            acc = [float(acc_data) for acc_data in lines[line*4+1].split()[1:]]
            self.data['acc'].append(acc)
            rot = [float(rot_data) for rot_data in lines[line*4+2].split()[1:]]
            self.data['rot'].append(rot)
            qua = [float(qua_data) for qua_data in lines[line*4+3].split()[1:]]
            self.data['qua'].append(qua)
        # print(self.time)
        # print(self.data)

    def preprocess_timing_gap(self):#计算数据点间隔的均值和方差，apple watch ≈ 0.01 ± 0.001 s
        timing_gap = []
        for i in range(len(self.time)-1):
            timing_gap.append(self.time[i+1]-self.time[i])
        timing_mean = np.mean(timing_gap)
        timing_std = np.std(timing_gap, ddof=1)
        print('mean = ', timing_mean)
        print('std  = ', timing_std)
        print('max  = ', np.max(timing_gap))
        print('min  = ', np.min(timing_gap))

    def show_single_plot(self):
        fig, axs = plt.subplots(10, 1)
        axs[0].plot(self.time, [data[0] for data in self.data['acc']])
        axs[1].plot(self.time, [data[1] for data in self.data['acc']])
        axs[2].plot(self.time, [data[2] for data in self.data['acc']])
        axs[3].plot(self.time, [data[0] for data in self.data['rot']])
        axs[4].plot(self.time, [data[1] for data in self.data['rot']])
        axs[5].plot(self.time, [data[2] for data in self.data['rot']])
        axs[6].plot(self.time, [data[0] for data in self.data['qua']])
        axs[7].plot(self.time, [data[1] for data in self.data['qua']])
        axs[8].plot(self.time, [data[2] for data in self.data['qua']])
        axs[9].plot(self.time, [data[3] for data in self.data['qua']])
        plt.show()

    def frequency_transform(self):
        self.acc_fft = [[], [], []]
        self.acc_fft_bucket = [[[],[],[]],[[],[],[]],[[],[],[]]]
        self.acc_fft_max = [[], [], []]
        self.acc_fft_max_time = []
        SAMPLE_LEN = 50
        SAMPLE_TIME = (int)(len(self.time)/SAMPLE_LEN)

        for i in range(SAMPLE_TIME):
            sample_data = self.data['acc'][i*SAMPLE_LEN: (i+1)*SAMPLE_LEN]
            sample = []
            sample_fft = []
            for j in range(3):
                sample.append([data[j] for data in sample_data])
                # print(len(sample))
                # print(sample[j])
                # 100Hz采样，取50个点fft，结果是100Hz频率的结果，
                sample_fft.append(abs(fft(sample[j])))
                # store the fft result
                for k in sample_fft[j]:
                    self.acc_fft[j].append(k)
                bucket = [0,0,0]
                for k in range((int)(SAMPLE_LEN/2)):
                    bucket[(int)((k+5)/10)] = bucket[(int)((k+5)/10)] + (sample_fft[j][k])
                    # if k < 5:
                    #     bucket[0] = bucket[0] + sample_fft[j][k]
                    # elif k < 10:
                    #     bucket[1] = bucket[1] + sample_fft[j][k]
                    # else:
                    #     bucket[2] = bucket[2] + sample_fft[j][k]
                for k in range(3):
                    self.acc_fft_bucket[j][k].append(bucket[k])
                # calculate max frequency
                # max_frequency = np.argmax(np.array(bucket))
                max_frequency = np.argmax(np.array(sample_fft[j][0:(int)(SAMPLE_LEN/2)]))
                self.acc_fft_max[j].append(max_frequency)
                # if max_frequency > 5:
                #     self.acc_fft_max[j].append(1)
                # else:
                #     self.acc_fft_max[j].append(0)
            self.acc_fft_max_time.append(i)

        #after processing
        self.acc_time = self.time[0:SAMPLE_LEN*SAMPLE_TIME]
        self.SAMPLE_LEN = SAMPLE_LEN
        self.SAMPLE_TIME = SAMPLE_TIME

    def frequency_show(self):
        fig, axs = plt.subplots(6, 1)
        axs[0].plot(self.time, [data[0] for data in self.data['acc']])
        axs[1].plot(self.acc_fft_max_time, self.acc_fft_max[0])
        axs[2].plot(self.time, [data[1] for data in self.data['acc']])
        axs[3].plot(self.acc_fft_max_time, self.acc_fft_max[1])
        axs[4].plot(self.time, [data[2] for data in self.data['acc']])
        axs[5].plot(self.acc_fft_max_time, self.acc_fft_max[2])
        plt.show()

    def frequency_bucket_show(self):
        fig, axs = plt.subplots(6, 1)
        axs[0].plot(self.time, [data[0] for data in self.data['acc']])
        axs[1].plot(self.acc_fft_max_time, self.acc_fft_bucket[0][1], self.acc_fft_max_time, self.acc_fft_bucket[0][2])
        axs[2].plot(self.time, [data[1] for data in self.data['acc']])
        axs[3].plot(self.acc_fft_max_time, self.acc_fft_bucket[1][1], self.acc_fft_max_time, self.acc_fft_bucket[1][2])
        axs[4].plot(self.time, [data[2] for data in self.data['acc']])
        axs[5].plot(self.acc_fft_max_time, self.acc_fft_bucket[2][1], self.acc_fft_max_time, self.acc_fft_bucket[2][2])
        plt.show()

    def find_peak_after_fft(self):
        self.peak_time = []
        THRESHOLD = 10
        for i in range(self.SAMPLE_TIME):
            isPeak = False
            #x,y,z轴中有高频能量
            if self.acc_fft_bucket[0][2][i] > THRESHOLD:
                isPeak = True
            elif self.acc_fft_bucket[1][2][i] > THRESHOLD:
                isPeak = True
            elif self.acc_fft_bucket[2][2][i] > THRESHOLD:
                isPeak = True

            if isPeak:
                start = self.acc_time[i*self.SAMPLE_LEN]
                end = self.acc_time[(i+1)*self.SAMPLE_LEN]
                self.peak_time.append([start, end])

def find_peak_index(data_unit):
    data_length = data_unit.shape[0]
    norm_unit = np.zeros((data_length, 2))
    for i in range(data_length):
        norm_unit[i, 0] = np.linalg.norm(data_unit[i, 0:3])
        norm_unit[i, 1] = np.linalg.norm(data_unit[i, 9:12])
    # print(norm_unit)
    return int (np.argmax(norm_unit) / 2) - 25