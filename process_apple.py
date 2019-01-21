import matplotlib.pyplot as plt
import math as math
import numpy as np
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
                    bucket[(int)(k/9)] = bucket[(int)(k/9)] + (sample_fft[j][k])
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
        axs[1].plot(self.acc_fft_max_time, self.acc_fft_bucket[0][0], self.acc_fft_max_time, self.acc_fft_bucket[0][1], self.acc_fft_max_time, self.acc_fft_bucket[0][2])
        axs[2].plot(self.time, [data[1] for data in self.data['acc']])
        axs[3].plot(self.acc_fft_max_time, self.acc_fft_bucket[1][0], self.acc_fft_max_time, self.acc_fft_bucket[1][1], self.acc_fft_max_time, self.acc_fft_bucket[1][2])
        axs[4].plot(self.time, [data[2] for data in self.data['acc']])
        axs[5].plot(self.acc_fft_max_time, self.acc_fft_bucket[2][0], self.acc_fft_max_time, self.acc_fft_bucket[2][1], self.acc_fft_max_time, self.acc_fft_bucket[2][2])
        plt.show()

    def find_peak_after_fft(self):
        self.peak_time = []
        THRESHOLD = 5
        for i in range(self.SAMPLE_TIME):
            for j in range(3):
                if self.acc_fft_max[j][i] > THRESHOLD:
                    #左闭右开
                    start = self.acc_time[i*self.SAMPLE_LEN]
                    end = self.acc_time[(i+1)*self.SAMPLE_LEN]
                    self.peak_time.append([start, end])
                    break

    def refine_acc(self):
        print('refine')


#initialize
left = Process('log-20190119-123108-WatchL.txt')
left.read_data()
left.preprocess_timing_gap()
# left.show_single_plot()
right = Process('log-20190119-123108-WatchR.txt')
right.read_data()
right.preprocess_timing_gap()
# right.show_single_plot()

#address the start timing gap
TIMING_DIFF = 5.58
right.time = [time+TIMING_DIFF for time in right.time]

#mix two datagram: original data visualization
# fig, axs = plt.subplots(9, 1)
# axs[0].plot(left.time, [data[0] for data in left.data['acc']], right.time, [data[0] for data in right.data['acc']])
# axs[1].plot(left.time, [data[1] for data in left.data['acc']], right.time, [data[1] for data in right.data['acc']])
# axs[2].plot(left.time, [data[2] for data in left.data['acc']], right.time, [data[2] for data in right.data['acc']])
# axs[3].plot(left.time, [data[0] for data in left.data['att']], right.time, [data[0] for data in right.data['att']])
# axs[4].plot(left.time, [data[1] for data in left.data['att']], right.time, [data[1] for data in right.data['att']])
# axs[5].plot(left.time, [data[2] for data in left.data['att']], right.time, [data[2] for data in right.data['att']])
# axs[6].plot(left.time, [data[0] for data in left.data['rot']], right.time, [data[0] for data in right.data['rot']])
# axs[7].plot(left.time, [data[1] for data in left.data['rot']], right.time, [data[1] for data in right.data['rot']])
# axs[8].plot(left.time, [data[2] for data in left.data['rot']], right.time, [data[2] for data in right.data['rot']])
# fig, axs = plt.subplots(3, 1)
# axs[0].plot(left.time, [data[0] for data in left.data['acc']])
# axs[1].plot(left.time, [data[1] for data in left.data['acc']])
# axs[2].plot(left.time, [data[2] for data in left.data['acc']])
# axs[3].plot(left.time, [data[0] for data in left.data['att']])
# axs[4].plot(left.time, [data[1] for data in left.data['att']])
# axs[5].plot(left.time, [data[2] for data in left.data['att']])
# axs[6].plot(left.time, [data[0] for data in left.data['rot']])
# axs[7].plot(left.time, [data[1] for data in left.data['rot']])
# axs[8].plot(left.time, [data[2] for data in left.data['rot']])
# plt.show()

# fig, axs = plt.subplots(3, 1)
# axs[0].plot(right.time, [data[0] for data in right.data['acc']])
# axs[1].plot(right.time, [data[1] for data in right.data['acc']])
# axs[2].plot(right.time, [data[2] for data in right.data['acc']])
# axs[3].plot(right.time, [data[0] for data in right.data['att']])
# axs[4].plot(right.time, [data[1] for data in right.data['att']])
# axs[5].plot(right.time, [data[2] for data in right.data['att']])
# axs[6].plot(right.time, [data[0] for data in right.data['rot']])
# axs[7].plot(right.time, [data[1] for data in right.data['rot']])
# axs[8].plot(right.time, [data[2] for data in right.data['rot']])
# plt.show()

#transform to frequency field by fft(sample n = 50 -> 50Hz, peek frequency = 20Hz, enough)
left.frequency_transform()
# left.frequency_show()
left.frequency_bucket_show()
right.frequency_transform()
# right.frequency_show()
right.frequency_bucket_show()

#find peak by data after fft
def overlap(x, y):
    if x[1] < y[0]:
        return -1
    elif x[0] > y[1]:
        return 1
    else:#overlap
        return 0

def peak_classification(t_left, t_right):
    left_start = left.time.index(t_left[0])
    left_end = left.time.index(t_left[1])
    right_start = right.time.index(t_right[0])
    right_end = right.time.index(t_right[1])
    #get acc data
    left_acc = np.array(left.data['acc'][left_start:left_end])
    right_acc = np.array(right.data['acc'][right_start:right_end])
    # print(np.shape(left_acc))
    # print(np.shape(right_acc))

    left_max = []
    right_max = []
    for i in range(3):
        left_max.append(np.max(abs(left_acc[:,i])))
        right_max.append(np.max(abs(right_acc[:,i])))
    print(left_max, right_max)
    left_coor = np.argmax(left_max)
    right_coor = np.argmax(right_max)
    print(left_coor, right_coor)
    # classification rules
    if left_coor == 2 and right_coor == 2:
        left_is_positive = False
        right_is_positive = False
        if left_max[2] in left_acc[:,2]:
            left_is_positive = True
        if right_max[2] in right_acc[:,2]:
            right_is_positive = True
        print(left_is_positive, right_is_positive)
        if left_is_positive == False and right_is_positive == False:
            print("palm to palm")
        elif left_is_positive == True and right_is_positive == False:
            print("back to back")
    elif left_coor == 0 and right_coor == 0:
        left_is_positive = False
        right_is_positive = False
        if left_max[0] in left_acc[:,0]:
            left_is_positive = True
        if right_max[0] in right_acc[:,0]:
            right_is_positive = True
        print(left_is_positive, right_is_positive)
        if left_is_positive == True and right_is_positive == False:
            print("fist to fist")





left.find_peak_after_fft()
right.find_peak_after_fft()
# find overlap peak
left_index = 0
right_index = 0
while left_index < len(left.peak_time) and right_index < len(right.peak_time):
    x = left.peak_time[left_index]
    y = right.peak_time[right_index]
    res = overlap(x, y)
    if res == -1:
        left_index = left_index + 1
    elif res == 1:
        right_index = right_index + 1
    else:#overlap
        print('find peak: left = [%f, %f], right = [%f, %f]'%(x[0], x[1], y[0], y[1]))
        peak_classification(x, y)#classification
        left_index = left_index + 1
        right_index = right_index + 1
left.frequency_show()
right.frequency_show()
