import matplotlib.pyplot as plt
import math as math
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.externals import joblib
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
        self.data = {'acc':[], 'att':[], 'rot':[]}

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
            att = [float(att_data) for att_data in lines[line*4+2].split()[1:]]
            att = [att[0], att[2], att[1]]### # WARNING:  new data format convert to old data # WARNING:
            self.data['att'].append(att)
            rot = [float(rot_data) for rot_data in lines[line*4+3].split()[1:]]
            self.data['rot'].append(rot)
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
        fig, axs = plt.subplots(9, 1)
        axs[0].plot(self.time, [data[0] for data in self.data['acc']])
        axs[1].plot(self.time, [data[1] for data in self.data['acc']])
        axs[2].plot(self.time, [data[2] for data in self.data['acc']])
        axs[3].plot(self.time, [data[0] for data in self.data['att']])
        axs[4].plot(self.time, [data[1] for data in self.data['att']])
        axs[5].plot(self.time, [data[2] for data in self.data['att']])
        axs[6].plot(self.time, [data[0] for data in self.data['rot']])
        axs[7].plot(self.time, [data[1] for data in self.data['rot']])
        axs[8].plot(self.time, [data[2] for data in self.data['rot']])
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
                    # bucket[(int)((k+5)/10)] = bucket[(int)((k+5)/10)] + (sample_fft[j][k])
                    if k < 5:
                        bucket[0] = bucket[0] + sample_fft[j][k]
                    elif k < 10:
                        bucket[1] = bucket[1] + sample_fft[j][k]
                    else:
                        bucket[2] = bucket[2] + sample_fft[j][k]
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
        # self.peak_time = []
        # THRESHOLD = 5
        # for i in range(self.SAMPLE_TIME):
        #     for j in range(3):
        #         if self.acc_fft_max[j][i] > THRESHOLD:
        #             #左闭右开
        #             start = self.acc_time[i*self.SAMPLE_LEN]
        #             end = self.acc_time[(i+1)*self.SAMPLE_LEN]
        #             self.peak_time.append([start, end])
        #             break

    def refine_acc(self):
        print('refine')


#initialize
left = Process('data/log-20190123-falsepalm-WatchL.txt')
left.read_data()
left.preprocess_timing_gap()
# left.show_single_plot()
right = Process('data/log-20190123-falsepalm-WatchR.txt')
right.read_data()
right.preprocess_timing_gap()
# right.show_single_plot()

#address the start timing gap
FILE_SHIFT = 0.064
TIMING_DIFF = left.time[0] - right.time[0]
right.time = [time+TIMING_DIFF-FILE_SHIFT for time in right.time]

#mix two datagram: original data visualization
fig, axs = plt.subplots(9, 1)
axs[0].plot(left.time, [data[0] for data in left.data['acc']], right.time, [data[0] for data in right.data['acc']])
axs[1].plot(left.time, [data[1] for data in left.data['acc']], right.time, [data[1] for data in right.data['acc']])
axs[2].plot(left.time, [data[2] for data in left.data['acc']], right.time, [data[2] for data in right.data['acc']])
axs[3].plot(left.time, [data[0] for data in left.data['att']], right.time, [data[0] for data in right.data['att']])
axs[4].plot(left.time, [data[1] for data in left.data['att']], right.time, [data[1] for data in right.data['att']])
axs[5].plot(left.time, [data[2] for data in left.data['att']], right.time, [data[2] for data in right.data['att']])
axs[6].plot(left.time, [data[0] for data in left.data['rot']], right.time, [data[0] for data in right.data['rot']])
axs[7].plot(left.time, [data[1] for data in left.data['rot']], right.time, [data[1] for data in right.data['rot']])
axs[8].plot(left.time, [data[2] for data in left.data['rot']], right.time, [data[2] for data in right.data['rot']])
plt.show()

clf = joblib.load("model/classification10_withnoise_model.m")

left_start = 0
right_start = 0
while abs(left.time[left_start] - right.time[right_start]) > 0.01:
    if left.time[left_start] < right.time[right_start]:
        left_start = left_start + 1
    else:
        right_start = right_start + 1
print('start_time: ', left_start, left.time[left_start], right_start, right.time[right_start])

def isAcc(k):
    if k >= 6 and k < 9:
        return True
    elif k >= 15 and k < 18:
        return True
    else:
        return False

length = 50
offset = 25
signal_array = []
while left_start + length < len(left.time) and right_start + length < len(right.time):
    store_data = []
    try:
        for k in range(length):
            for j in range(3):
                store_data.append(left.data['acc'][left_start+k][j])
            for j in range(3):
                store_data.append(left.data['att'][left_start+k][j])
            for j in range(3):
                store_data.append(left.data['rot'][left_start+k][j])
            for j in range(3):
                store_data.append(right.data['acc'][right_start+k][j])
            for j in range(3):
                store_data.append(right.data['att'][right_start+k][j])
            for j in range(3):
                store_data.append(right.data['rot'][right_start+k][j])

        data_unit = (np.array(store_data)).reshape(50, 18)
        #fft first
        is_signal = False
        for k in range(3):
            fft_unit = np.array(abs(fft(data_unit[:, k])))
            if np.sum(fft_unit[5:15]) > 10:
                is_signal = True
                break
            fft_unit = np.array(abs(fft(data_unit[:, 9+k])))
            if np.sum(fft_unit[5:15]) > 10:
                is_signal = True
                break
        if not is_signal:
            signal_array.append(7)
        else:
            #if frequency energy is high enough, judge signal
            feature_length = 48
            featured_unit = np.zeros((feature_length))
            for k in range(18):
                if not isAcc(k):
                    data_unit_coor = data_unit[:, k]
                    if k >= 9:
                        k = k - 3
                    featured_unit[4*k] = np.min(data_unit_coor)
                    featured_unit[4*k+1] = np.max(data_unit_coor)
                    featured_unit[4*k+2] = np.mean(data_unit_coor)
                    featured_unit[4*k+3] = np.std(data_unit_coor)
            res = clf.predict([featured_unit])
            # print(res[0])
            signal_array.append(res[0])
            # if res[0] == 0:
            #     display_data = np.array(store_data).reshape(-1, 18)
            #     display_index = np.arange(50)
            #     print(display_data.shape)
            #     fig, axs = plt.subplots(9, 1)
            #     for i in range(9):
            #         axs[i].plot(display_index, display_data[:, i], display_index, display_data[:, 9+i])
            #     plt.show()
    except:
        # print(store_data)
        print('NaN error')
        signal_array.append(7)
        break
    finally:
        left_start = left_start + offset
        right_start = right_start + offset
        if abs(left.time[left_start] - right.time[right_start]) > 0.01:
            print('start_time: ', left_start, left.time[left_start], right_start, right.time[right_start])
            while abs(left.time[left_start] - right.time[right_start]) > 0.01:
                if left.time[left_start] < right.time[right_start]:
                    left_start = left_start + 1
                else:
                    right_start = right_start + 1

fig, axs = plt.subplots(4, 1)
axs[0].plot(left.time, [data[0] for data in left.data['acc']], right.time, [data[0] for data in right.data['acc']])
axs[1].plot(left.time, [data[1] for data in left.data['acc']], right.time, [data[1] for data in right.data['acc']])
axs[2].plot(left.time, [data[2] for data in left.data['acc']], right.time, [data[2] for data in right.data['acc']])
axs[3].plot(signal_array)
plt.show()
