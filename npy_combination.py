import numpy as np

file1 = 'training/sound/hbj/PxP_np.npy'
file2 = 'training/sound/hbj/PxP2_np.npy'

data1 = np.load(file1)
data2 = np.load(file2)
print(data1.shape, data2.shape)

data_comb = np.concatenate((data1, data2))
print(data_comb.shape)
np.save('training/sound/hbj/PxPP_np.npy', data_comb)
