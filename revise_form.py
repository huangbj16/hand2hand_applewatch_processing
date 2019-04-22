import os
import numpy as np

gestures_list = []
gestures_type = []

rootdir = 'data/sound_final/yyw/'
list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
print(list)
for filename in list:
    if not '.txt' in filename:
        continue
    print(filename)
    path = os.path.join(rootdir, filename)
    alter_path = os.path.join('data/sound_final/alter/', filename)
    f = open(path, 'r')
    lines = f.readlines()
    f.close()
    g = open(alter_path, 'w')
    for line in lines:
        if not 'att' in line:
            g.write(line)
    g.close()