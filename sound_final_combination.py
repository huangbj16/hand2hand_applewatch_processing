import os
import numpy as np

suffixes = ['lyq/', 'yzc/', 'yyk/', 'yyw/', 'swn/', 'sy/', 'lgh/', 'ycy/']
for suffix in suffixes:
    gestures_list = []
    gestures_type = []
    rootdir = 'training/sound_final/'+suffix
    for i in range(10):
        list = os.listdir(rootdir+'swipe_'+str(i)+'/') #列出文件夹下所有的目录与文件
        print(list)
        for filename in list:
            path = os.path.join(rootdir,'swipe_'+str(i)+'/'+filename)
            data = np.load(path)
            gesture_type = int(filename[0])
            if gesture_type in gestures_type:
                index = gestures_type.index(gesture_type)
                print(gestures_list[index].shape, data.shape)
                if data.shape[0] != 0:
                    gestures_list[index] = np.concatenate((gestures_list[index], data))
                print(gestures_list[index].shape)
            else:
                gestures_type.append(gesture_type)
                gestures_list.append(data)

    for i in range(len(gestures_type)):
        print(gestures_type[i], gestures_list[i].shape)
        np.save(rootdir+'combination/'+'swipe_'+str(gestures_type[i])+'_np', gestures_list[i])