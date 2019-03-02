import matplotlib.pyplot as plt
import math as math
import numpy as np
import random
from scipy.fftpack import fft,ifft

''' 流程
修改文件名
计算误差
修改提取样本的时域频域阈值
采集样本
'''

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
left = Process('data/motion/log-20190218-IyG-WatchL.txt')
left.read_data()
left.preprocess_timing_gap()
# left.show_single_plot()
right = Process('data/motion/log-20190218-IyG-WatchR.txt')
right.read_data()
right.preprocess_timing_gap()
# right.show_single_plot()

# print(len(left.time))
# for i in range(len(left.time)):
#     if left.time[i] > left.time[i+1]:
#         print(i)
#         print(len(left.time))
#         j = i
#         while j < len(left.time):
#             print(left.time[j])
#             j = j+1

#address the start timing gap
FILE_SHIFT = -2.17
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

left_start = 0
right_start = 0
while abs(left.time[left_start] - right.time[right_start]) > 0.01:
    if left.time[left_start] < right.time[right_start]:
        left_start = left_start + 1
    else:
        right_start = right_start + 1
print('start_time: ', left_start, left.time[left_start], right_start, right.time[right_start])


def find_peak_index(data_unit):
    data_length = data_unit.shape[0]
    norm_unit = np.zeros((data_length, 2))
    for i in range(data_length):
        norm_unit[i, 0] = np.linalg.norm(data_unit[i, 0:3])
        norm_unit[i, 1] = np.linalg.norm(data_unit[i, 9:12])
    # print(norm_unit)
    return int (np.argmax(norm_unit) / 2) - 25

length = 50
cover_array = []
fft_cover_array = []
count = 0
fft_count = 0
both_count = 0
THRESHOLD = 6#change
TIME_THRESHOLD = 0.6#change
index_array = np.arange(0, 50)
store_data_list = []

while left_start + length < len(left.time) and right_start + length < len(right.time):
    left_data = (np.array(left.data['acc'][left_start: left_start+length])).T
    right_data = (np.array(right.data['acc'][right_start: right_start+length])).T
    # print(left_data.shape, right_data.shape)
    #time field is peak
    is_time_peak = False
    for i in range(3):
        if np.max(np.fabs(left_data[i])) > TIME_THRESHOLD or np.max(np.fabs(right_data[i])) > TIME_THRESHOLD:#change
            is_time_peak = True
            break
    if is_time_peak:
        cover_array.append(1)
        count = count + 1
    else:
        cover_array.append(0)

    #frequency field is peak
    left_data_fft = np.zeros(left_data.shape)
    right_data_fft = np.zeros(right_data.shape)
    for i in range(3):
        left_data_fft[i] = np.array(abs(fft(left_data[i])))
        right_data_fft[i] = np.array(abs(fft(right_data[i])))
    left_bucket = [[0,0,0], [0,0,0], [0,0,0]]
    right_bucket = [[0,0,0], [0,0,0], [0,0,0]]
    for j in range(3):
        for k in range((int)(length/2)):
            left_bucket[j][(int)((k+5)/10)] = left_bucket[j][(int)((k+5)/10)] + (left_data_fft[j, k])
            right_bucket[j][(int)((k+5)/10)] = right_bucket[j][(int)((k+5)/10)] + (right_data_fft[j, k])
    is_peak = False
    for j in range(3):
        if left_bucket[i][1] > THRESHOLD or right_bucket[i][1] > THRESHOLD:#change 现在取的是5-15hz的能量值判断。
            is_peak = True
        break
    if is_peak:
        fft_cover_array.append(1)
        fft_count = fft_count + 1
    else:
        fft_cover_array.append(0)

    if is_peak and is_time_peak:
        #double confirm, no align currently
        both_count = both_count + 1
        #align
        # peak_index = np.argmax(np.fabs(right_data[1])) - 25 - int(random.random()*20-10)#change
        # left_start = left_start + peak_index
        # right_start = right_start + peak_index
        left_data = (np.array(left.data['acc'][left_start: left_start+length])).T
        right_data = (np.array(right.data['acc'][right_start: right_start+length])).T
        # fig, axs = plt.subplots(3, 1)
        # for i in range(3):
        #     axs[i].plot(index_array, left_data[i], index_array, right_data[i])
        # plt.show()
        #store data
        store_data = []
        for i in range(length):
            for j in range(3):
                store_data.append(left.data['acc'][left_start+i][j])
            for j in range(3):
                store_data.append(left.data['att'][left_start+i][j])
            for j in range(3):
                store_data.append(left.data['rot'][left_start+i][j])
            for j in range(3):
                store_data.append(right.data['acc'][right_start+i][j])
            for j in range(3):
                store_data.append(right.data['att'][right_start+i][j])
            for j in range(3):
                store_data.append(right.data['rot'][right_start+i][j])
        data_unit = (np.array(store_data)).reshape(50, 18)
        peak_index = find_peak_index(data_unit) + int(random.random()*20-10)
        left_start = left_start + peak_index
        right_start = right_start + peak_index
        store_data = []
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
        store_data_list.append(store_data)
    #move to next
    left_start = left_start + length
    right_start = right_start + length
    if abs(left.time[left_start] - right.time[right_start]) > 0.01:
        print('start_time: ', left_start, left.time[left_start], right_start, right.time[right_start])
        while abs(left.time[left_start] - right.time[right_start]) > 0.01:
            if left.time[left_start] < right.time[right_start]:
                left_start = left_start + 1
            else:
                right_start = right_start + 1

store_data_list = np.array(store_data_list)[5:]
print('data size: ', store_data_list.shape)
np.save('training/motion/log-20190218-IyG-WatchL_np', store_data_list)

fig, axs = plt.subplots(5, 1)
axs[0].plot(left.time, [data[0] for data in left.data['acc']], right.time, [data[0] for data in right.data['acc']])
axs[1].plot(left.time, [data[1] for data in left.data['acc']], right.time, [data[1] for data in right.data['acc']])
axs[2].plot(left.time, [data[2] for data in left.data['acc']], right.time, [data[2] for data in right.data['acc']])
axs[3].plot(cover_array)
axs[4].plot(fft_cover_array)
plt.show()

print(count, fft_count, both_count)

for i in range(store_data_list.shape[0]):
    data_unit = store_data_list[i].reshape(50, 18)
    dimension = 18
    
    fig, axs = plt.subplots(9, 2)
    plt.setp(axs, ylim=(-5, 5))

    for j in range(dimension):
        # if j < 3:
        #     axs[j%3][int(j/3)].plot(data_unit[:, j])
        # else:
        #     axs[j%3][int(j/3)].plot(data_unit[:, j+6])
        axs[j%9][int(j/9)].plot(data_unit[:, j])
    
    plt.show()


#find peak by data after fft
# def overlap(x, y):
#     if x[1] < y[0]:
#         return -1
#     elif x[0] > y[1]:
#         return 1
#     else:#overlap
#         return 0

# def peak_classification(t_left, t_right):
#     left_start = left.time.index(t_left[0])
#     left_end = left.time.index(t_left[1])
#     right_start = right.time.index(t_right[0])
#     right_end = right.time.index(t_right[1])
#     #get acc data
#     left_acc = np.array(left.data['acc'][left_start:left_end])
#     right_acc = np.array(right.data['acc'][right_start:right_end])
#     # print(np.shape(left_acc))
#     # print(np.shape(right_acc))

#     left_max = []
#     right_max = []
#     for i in range(3):
#         left_max.append(np.max(abs(left_acc[:,i])))
#         right_max.append(np.max(abs(right_acc[:,i])))
#     print(left_max, right_max)
#     left_coor = np.argmax(left_max)
#     right_coor = np.argmax(right_max)
#     print(left_coor, right_coor)
#     # classification rules
#     if left_coor == 2 and right_coor == 2:
#         left_is_positive = False
#         right_is_positive = False
#         if left_max[2] in left_acc[:,2]:
#             left_is_positive = True
#         if right_max[2] in right_acc[:,2]:
#             right_is_positive = True
#         print(left_is_positive, right_is_positive)
#         if left_is_positive == False and right_is_positive == False:
#             print("palm to palm")
#         elif left_is_positive == True and right_is_positive == False:
#             print("back to back")
#     elif left_coor == 0 and right_coor == 0:
#         left_is_positive = False
#         right_is_positive = False
#         if left_max[0] in left_acc[:,0]:
#             left_is_positive = True
#         if right_max[0] in right_acc[:,0]:
#             right_is_positive = True
#         print(left_is_positive, right_is_positive)
#         if left_is_positive == True and right_is_positive == False:
#             print("fist to fist")

# left.find_peak_after_fft()
# right.find_peak_after_fft()
# # find overlap peak
# left_index = 0
# right_index = 0
# overlap_count = 0
# #display use array
# # index_array = np.arange(0, 50)
# cover_array = np.zeros(len(left.time))
# #store file
# training_filename = 'training/motion/fist.txt'
# write_file = open(training_filename, 'w')
# #read peak from file
# read_file = open('data/fist-peak.txt', 'r')
# write_array = []

# while left_index < len(left.peak_time) and right_index < len(right.peak_time):
#     x = left.peak_time[left_index]
#     y = right.peak_time[right_index]
#     res = overlap(x, y)
#     if res == -1:
#         left_index = left_index + 1
#     elif res == 1:
#         right_index = right_index + 1
#     else:#overlap
#         print('find peak: left = [%f, %f], right = [%f, %f]'%(x[0], x[1], y[0], y[1]))
#         overlap_count = overlap_count + 1
#         #display coverage
#         left_time_index = [left.time.index(x[0]), left.time.index(x[1])]
#         right_time_index = [right.time.index(y[0]), right.time.index(y[1])]
#         for index in range(left_time_index[0], left_time_index[1]):
#             cover_array[index] = 1
#         # #display
#         # fig, axs = plt.subplots(3, 1)
#         # left_time = left.time[left_time_index[0]:left_time_index[1]]
#         # right_time = right.time[right_time_index[0]: right_time_index[1]]
#         # left_data = left.data['acc'][left_time_index[0]: left_time_index[1]]
#         # right_data = right.data['acc'][right_time_index[0]: right_time_index[1]]
#         # axs[0].plot(left_time, [data[0] for data in left_data], right_time, [data[0] for data in right_data])
#         # axs[1].plot(left_time, [data[1] for data in left_data], right_time, [data[1] for data in right_data])
#         # axs[2].plot(left_time, [data[2] for data in left_data], right_time, [data[2] for data in right_data])
#         # plt.show()
#         # # continue in advance
#         # left_index = left_index + 1
#         # right_index = right_index + 1
#         # continue

#         #wait for input to decide the shift
#         # array_peak = input("input where is the peak: ")
#         # array_peak = str(array_peak)
#         # array_peak = [float(array_peak), float(array_peak)+0.2]

#         ####use
#         peak = read_file.readline()
#         if 'no' in peak:
#             left_index = left_index + 1
#             right_index = right_index + 1
#             continue
#         else:
#             peak = float(peak)
#         array_peak = [peak, peak+0.2]
#         print(array_peak)
#         while left.time[left_time_index[0]] < array_peak[0]:
#             left_time_index[0] = left_time_index[0] + 1
#         while left.time[left_time_index[0]] > array_peak[0]:
#             left_time_index[0] = left_time_index[0] - 1
#         while left.time[left_time_index[1]] > array_peak[1]:
#             left_time_index[1] = left_time_index[1] - 1
#         while left.time[left_time_index[1]] < array_peak[1]:
#             left_time_index[1] = left_time_index[1] + 1
#         while right.time[right_time_index[0]] < array_peak[0]:
#             right_time_index[0] = right_time_index[0] + 1
#         while right.time[right_time_index[0]] > array_peak[0]:
#             right_time_index[0] = right_time_index[0] - 1
#         while right.time[right_time_index[1]] > array_peak[1]:
#             right_time_index[1] = right_time_index[1] - 1
#         while right.time[right_time_index[1]] < array_peak[1]:
#             right_time_index[1] = right_time_index[1] + 1
#         #now bound is at the edge of peak

#         #store data
#         length = 50
#         left_time_index[1] = left_time_index[1] - length
#         right_time_index[1] = right_time_index[1] - length
#         while left_time_index[1] < left_time_index[0] and right_time_index[1] < right_time_index[0]:
#             store_data = []
#             for i in range(length):
#                 for j in range(3):
#                     store_data.append(left.data['acc'][left_time_index[1]+i][j])
#                 for j in range(3):
#                     store_data.append(left.data['att'][left_time_index[1]+i][j])
#                 for j in range(3):
#                     store_data.append(left.data['rot'][left_time_index[1]+i][j])
#                 for j in range(3):
#                     store_data.append(right.data['acc'][right_time_index[1]+i][j])
#                 for j in range(3):
#                     store_data.append(right.data['att'][right_time_index[1]+i][j])
#                 for j in range(3):
#                     store_data.append(right.data['rot'][right_time_index[1]+i][j])
#             write_file.write('3 ')#1 means palm
#             print(len(store_data))
#             write_array.append(store_data)
#             for data in store_data:
#                 write_file.write(str(data)+' ')
#             write_file.write('\n')
#             # fig, axs = plt.subplots(3, 1)
#             # left_time = left.time[left_time_index[1]: left_time_index[1]+length]
#             # right_time = right.time[left_time_index[1]: left_time_index[1]+length]
#             # left_data = left.data['acc'][left_time_index[1]: left_time_index[1]+length]
#             # right_data = right.data['acc'][left_time_index[1]: left_time_index[1]+length]
#             # axs[0].plot(left_time, [data[0] for data in left_data], right_time, [data[0] for data in right_data])
#             # axs[1].plot(left_time, [data[1] for data in left_data], right_time, [data[1] for data in right_data])
#             # axs[2].plot(left_time, [data[2] for data in left_data], right_time, [data[2] for data in right_data])
#             # plt.show()
#             left_time_index[1] = left_time_index[1] + 1
#             right_time_index[1] = right_time_index[1] + 1

#         #move to next peak
#         left_index = left_index + 1
#         right_index = right_index + 1
# print(overlap_count)
# # left.frequency_show()
# # right.frequency_show()
# fig, axs = plt.subplots(4, 1)
# axs[0].plot(left.time, [data[0] for data in left.data['acc']], right.time, [data[0] for data in right.data['acc']])
# axs[1].plot(left.time, [data[1] for data in left.data['acc']], right.time, [data[1] for data in right.data['acc']])
# axs[2].plot(left.time, [data[2] for data in left.data['acc']], right.time, [data[2] for data in right.data['acc']])
# axs[3].plot(left.time, cover_array)
# plt.show()

# write_array = np.array(write_array)
# print(np.shape(write_array))
# np.save('training/fist_np', write_array)

# write_file.close()
# read_file.close()
