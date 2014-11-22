
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import sys
import os


def nnr(datapath):
	# load mat
	datafile = os.path.join(datapath, 'data_numpy.mat')
	if os.path.exists(datafile) is False:
		print('Data file %s not found.' % datafile)

	data_numpy = sio.loadmat(datafile)
	# get training and test data
	train_x = data_numpy['trainX'];
	train_xn= data_numpy['trainXN']; 	# normalized x
	train_y = data_numpy['trainY'];
	test_x  = data_numpy['testX'];
	test_xn = data_numpy['testXN'];		# normalized x
	test_y  = data_numpy['testY'];
	base_y  = data_numpy['baseY'];

	# visualize one data
	# index = 10
	# x = list(xrange(len(train_x[index])))
	# y = [train_y[index][0] for i in x]
	# plt.plot(x, train_x[index])
	# plt.plot(x, y)
	# plt.show()
	neighbor_num = 5
	neigh = KNeighborsRegressor(n_neighbors=neighbor_num,weights='distance')
	neigh.fit(train_x, train_y)
	pred_y = neigh.predict(test_x)
	np.savetxt(os.path.join(datapath, 'nnr_pred.txt'), pred_y)
	# print test_y
	# print pred_y

	## plot results
	#max_id = 1000
	#x = list(xrange(len(test_y[1:max_id])))
	#h_truth = plt.plot(x, test_y[1:max_id], color = 'green',label='Ground truth')
	#h_pred  = plt.plot(x, pred_y[1:max_id], color = 'red', label='Lasso')
	#h_base  = plt.plot(x, base_y[1:max_id], color = 'blue', label='[1]')
	##plt.legend(handles = [h_truth, h_pred, h_base])
	#plt.show()


if __name__ == '__main__':
	if len(sys.argv) < 2:
		print("Usage: nnr.py datapath")
		exit()
	else:
		nnr(sys.argv[1])
