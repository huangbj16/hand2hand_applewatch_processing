import numpy as np

file_to_convert = ['palm', 'back', 'fist']

for i in range(3):
    filename = 'training/' + file_to_convert[i] + '_np.npy'
    old_data = np.load(filename)
    print(old_data.shape)
    #take 1 row out of 50 rows, swap att[1] and att[2]
    new_data = old_data[24::25]
    print(new_data.shape)
