import os
import numpy as np

gestures_list = []
gestures_type = []

suffixes = ['hbj', 'lyq', 'yzc', 'fjy', 'yyk', 'yyw', 'swn', 'sy', 'lgh', 'ycy']

for suffix in suffixes:
    rootdir = 'data/sound_final/swipe/'+suffix+'-swipe/'
    list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
    print(list)
    for filename in list:
        if not '.txt' in filename:
            continue
        print(filename)
        path = os.path.join(rootdir, filename)
        # alter_path = os.path.join('data/sound_final/swipe/fjy-swipe/', filename)
        f = open(path, 'r')
        lines = f.readlines()
        f.close()
        g = open(path, 'w')
        for line in lines:
            if not 'att' in line:
                g.write(line)
        g.close()