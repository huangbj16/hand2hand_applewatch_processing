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

加入声音处理，使用mfcc局部过程特征。

在pilot study中，4人8分类 达到90左右的within-accuracy和70左右的independent-accuracy。

**Special thanks to python package python-speech-feature**

By Bingjian Huang

Copyrights Reserved