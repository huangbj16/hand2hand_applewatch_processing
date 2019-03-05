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

By Bingjian Huang


