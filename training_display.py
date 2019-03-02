import numpy as np
import matplotlib.pyplot as plt
import os

type_array = []

motion_type = []

rootdir = 'training/motion/'
list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
for i in range(0, len(list)):
    motion_type.append(list[i])
    path = os.path.join(rootdir,list[i])
    print(path)
    data = np.load(path)
    print(data.shape)
    type_array.append(data)
print(len(type_array))

rootdir = 'training/motion_lu/'
list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
for i in range(0, len(list)):
    path = os.path.join(rootdir,list[i])
    print(path)
    data = np.load(path)
    print(data.shape)
    if list[i] in motion_type:
        mark = motion_type.index(list[i])
        previous_data = type_array[mark]
        type_array[mark] = np.concatenate((previous_data, data))
    else:
        type_array.append(data)
        motion_type.append(list[i])

print(len(type_array))

display_type = 13
print('display: ', motion_type[display_type])

data_length = type_array[display_type].shape[0]
primitive_data = type_array[display_type]

for i in range(data_length):
    data_unit = primitive_data[i].reshape(50, 18)
    dimension = 18
    
    fig, axs = plt.subplots(9, 2)
    plt.setp(axs, ylim=(-1, 1))

    for j in range(dimension):
        # if j < 3:
        #     axs[j%3][int(j/3)].plot(data_unit[:, j])
        # else:
        #     axs[j%3][int(j/3)].plot(data_unit[:, j+6])
        axs[j%9][int(j/9)].plot(data_unit[:, j])
    
    plt.show()


