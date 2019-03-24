# hand2hand_applewatch_processing

A set of programs aiming to detect and recognize hand2hand motions.

SOTA:

8 gestures subset, svm accuracy achieved 98%.

with complete offline test and sklearn & opencv svm training code

total 24 gestures set

24分类+降数据精度后，准确率仍然能达到98%
在offline测试中，
IyP left/up 与 噪声无法区分；
IyP right / IyP left / IyP up / PyP rigth无法区分，推测是因为att相同，acc变化过小或无规律（而且在从原始数据截取手势时，因为没有明确的对齐标志，可能对齐有问题；
IxP 与 IyP up无法区分，推测是因为att相同，并且做IyP up时，因为动作不连贯，可能产生一个和IxP相同的z轴方向的冲击；
IxFG（食指敲击指根关节）/ IxB / PxB无法区分，三者att基本相同，acc也相似。

下一步准备尝试更改feature，看能不能用带sync的feature，比如corr之类的。

2019/3/2 log
增加数据展示功能，用于通过肉眼观察不同动作在信号上的区别。
观察得知，IyP的动作acc确实太小。
IxFG（食指敲击指根关节）/ IxB / PxB 在acc和rot上的变化极其相似，区别可能是峰值的幅度。

2019/3/5 log
增加声音信息，正在尝试发现不同动作声音的区分度。
动作和噪声在频域上区别很大，现在的难题是如何和传感器同步提取出声音信息。
跑步中的噪声非常大，频域上信号非常复杂。
IyP四个方向的滑动，声音信号非常弱(<0.01)，频域能量也非常低。
测试MFCC，发现很好用，可以发现IyP信号的特征，很厉害！
目前发现的声音的作用：主要是增强PxP等原本已识别的手势，有望区分PxP和IxP，有望区分IyP和噪声。
遇到问题：左右手的sensor和audio四个数组如何对齐？
进展：初步完成了对齐。

2019/3/9 log
今天希望能完成数据切割，测试正确率。
发现采集程序的一个bug，sensor和audio时间长度不同，audio提前停了，原因不明，需要汇报。

2019/3/12 log
audio analysis 基本完成，发现了未对齐的问题，需要解决。
论文方案出现了路线性分歧，design or technique?
手动查看数据内容，分析acc和att

2019/3/13 log
已解决audio analysis的问题。

2019/3/16 log
学长给的数据没有auto_align……gg……
要写手动对齐的代码，同时发现了原来对齐中出现的问题。
增加了基于起始点的对齐方式，得到三组数据。

尝试通过freq_audio来分类，通过mfcc来分类。
分类效果皆不佳，开始找原因。
把IxB猜成了PxB

2019/3/18 log
继续查看声音特征
手动查看了声音特征，取了时域数据的min max mean std，而份额里准确率到95%
频率Bucket，无效果，改进方向：log bucket。
提取了freq argrelmax 即极大值频率，作为分类依据，准确率max min mean = : 0.9487179487179487 0.6153846153846154 0.8253846153846154

采集数据：2*PxP FDxP DxP mucus FDxB DxB
记得以后采数据要先拍一下手！！！autoalign！！！

2019/3/19 log
昨天采集并处理完了数据，
今天仍然使用了find peak的方法分类，发现效果很一般。
查阅了一些论文，如
    Design and Implementation of Frequency Domain Feature Extraction Module for Human Motion Recognition
    AMP: a new time-frequency feature extraction method for intermittent time-series data
得到了一些处理的灵感，尝试了STFT，发现有点意思，说不定能有效果。

接下来尝试log bucket
log bucket：限制在40个feature以内，一只手20个，log均分到5000，0 10 20 40 80 160 320 640 1280 2560 5120
归一化之后，
1-norm仍然很差，
但是2-norm效果特别好，7分类max min mean = : 0.9333333333333333 0.7523809523809524 0.8477142857142858
通过查看confusion matrix，发现容易混淆的手势为D和FD在各个面的碰撞，以及无法解释的FDxP和PxB
尝试 np.mean+std，不行，频域上出现了负数的能量。
尝试用一个人的数据预测另一个人的，3分类0.56，不合理。


2019/3/22 log
看其它声音论文是否有使用频域上的特征？？
今天读了十篇声音相关的论文，总结了哪些feature可以使用。

2019/3/24 log
尝试之前总结的feature
模仿了Detecting论文，使用了maximum, minimum, and average values of pitch, flux, roll-off, centroid, ZCR, RMS, and SNR的特征，效果一般。
接下来尝试mfcc和stft。

尝试STFT：如何使用STFT的二维矩阵信息？


如何解决双击的识别？

By Bingjian Huang

Copyrights Reserved