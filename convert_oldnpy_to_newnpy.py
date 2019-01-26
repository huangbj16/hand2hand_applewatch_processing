import numpy as np

file_to_convert = ['noise']

for i in range(1):
    filename = 'training/' + file_to_convert[i] + '_np.npy'
    old_data = np.load(filename)
    print(old_data.shape)
    #swap att[1] and att[2]
    new_data = old_data
    new_data[:, 4] = old_data[:, 5]
    new_data[:, 5] = old_data[:, 4]
    new_data[:, 13] = old_data[:, 14]
    new_data[:, 14] = old_data[:, 13]
    print(new_data.shape)
    np.save('training/' + file_to_convert[i] + '_new_np.npy', new_data)
    print(old_data[:, 4])
    print(new_data[:, 5])
