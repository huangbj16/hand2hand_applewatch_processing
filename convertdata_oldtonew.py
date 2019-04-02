import numpy as np
import matplotlib.pylab as plt
from sensor_module import Process

is_display_on = False

#initialize
left_sensor = Process('data/sound/hbj/log-20190325-IyPup-WatchL.txt')
left_sensor.read_data()
left_sensor.preprocess_timing_gap()
# left_sensor.show_single_plot()
right_sensor = Process('data/sound/hbj/log-20190325-IyPup-WatchR.txt')
right_sensor.read_data()
right_sensor.preprocess_timing_gap()
# right_sensor.show_single_plot()

primitive_data = np.load('training/sound/hbj/IyPup_np.npy')

data_length = primitive_data.shape[0]

new_data = np.zeros((data_length, 45100))

for j in range(data_length):
    segment = primitive_data[j]
    data_unit = segment[0:900].reshape(50, 18)
    audio_left = segment[900:900+22050]
    audio_right = segment[900+22050:900+44100]
    
    segment_primitive = data_unit[:, 0]
    x_max = np.max(segment_primitive)
    x_argmax = np.argmax(segment_primitive)
    data = np.array(left_sensor.data['acc'])[:, 0]
    location = np.argwhere(data == x_max)
    print('left:', location)
    if len(location) != 1:
        exit(0)
    location_left = location[0][0]-x_argmax

    segment_primitive = data_unit[:, 9]
    x_max = np.max(segment_primitive)
    x_argmax = np.argmax(segment_primitive)
    data = np.array(right_sensor.data['acc'])[:, 0]
    location = np.argwhere(data == x_max)
    print('right:', location)
    if len(location) != 1:
        exit(0)
    location_right = location[0][0]-x_argmax

    qua_left = ((np.array(left_sensor.data['qua']))[:, 3])[location_left:location_left+50]
    qua_right = ((np.array(right_sensor.data['qua']))[:, 3])[location_right:location_right+50]

    data_unit_new = np.zeros((50, 20))
    data_unit_new[:, 0:9] = data_unit[:, 0:9]
    data_unit_new[:, 9] = qua_left
    data_unit_new[:, 10:19] = data_unit[:, 9:18]
    data_unit_new[:, 19] = qua_right
    segment_new = np.concatenate((data_unit_new.reshape(-1), audio_left, audio_right))
    new_data[j] = segment_new

    if is_display_on:
        fig, axs = plt.subplots(10, 2)
        for i in range(20):
            axs[int(i%10), int(i/10)].plot(data_unit_new[:, i])
        plt.show()

np.save('training/sound_new/hbj/IyPup_np', new_data)


    
    
    


    



