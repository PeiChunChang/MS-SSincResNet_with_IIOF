import numpy as np
import os

path = '/media/maplepig/Data2/Datasets/PMEmo/CV_10_with_val/'

data_list = os.listdir(path)

for item in data_list:
	print(item)
	data = np.load(path + item)
	data = np.char.replace(data, 'npy/', '', 1)
	np.save(path + item, data)
	# print(data)
	# stop()