import numpy as np

f = open('training/back.txt', 'r')
lines = f.readlines()
array = []
for line in lines:
    subarray = []
    conts = line.split()
    for i in range(1, len(conts)):
        subarray.append(float(conts[i]))
    array.append(subarray)
array = np.array(array)
print(np.shape(array))
np.save('training/back_np', array)
