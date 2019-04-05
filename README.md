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

####纯IMU
dependent classification
左手c8：max min mean = : 0.9380804953560371 0.8452012383900929 0.8973374613003097
右手c8：max min mean = : 0.9195046439628483 0.8111455108359134 0.8612693498452012
双手c8：max min mean = : 0.9876160990712074 0.9473684210526315 0.9716099071207432

independent classification
左手c8：[0.4075, 0.3292079207920792, 0.582716049382716, 0.48267326732673266]
右手c8：[0.4475, 0.3415841584158416, 0.44938271604938274, 0.37376237623762376]
双手c8：[0.605, 0.6732673267326733, 0.7555555555555555, 0.6584158415841584]

####IMU+SOUND
dependent classification
左手c8：max min mean = : 0.9907120743034056 0.9473684210526315 0.9687925696594428
右手c8：max min mean = : 0.9907120743034056 0.9380804953560371 0.9638390092879259
双手c8：max min mean = : 1.0 0.9628482972136223 0.9826625386996903

independent classification
左手c8：[0.63, 0.6534653465346535, 0.7111111111111111, 0.693069306930693]
右手c8：[0.6675, 0.6881188118811881, 0.7135802469135802, 0.7821782178217822]
双手c8：[0.71, 0.745049504950495, 0.7679012345679013, 0.8514851485148515]

**Special thanks to python package python-speech-feature**

By Bingjian Huang

Copyrights Reserved