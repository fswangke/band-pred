print(__doc__)

from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os
import sys


def enet(datapath):
	# load data
	datafile = os.path.join(datapath, 'data_numpy.mat')
	if os.path.exists(datafile) is False:
		print('Data file %s not found.' % datafile)
		return

	data_numpy = sio.loadmat(datafile)
	train_x = data_numpy['trainX'];
	train_xn= data_numpy['trainXN']; 	# normalized x
	train_y = data_numpy['trainY'];
	test_x  = data_numpy['testX'];
	test_xn = data_numpy['testXN'];		# normalized x
	test_y  = data_numpy['testY'];
	base_y  = data_numpy['baseY'];

	train_y = train_y.ravel()
	enet = ElasticNet(alpha=0.5, l1_ratio=0.7, max_iter=10000)
	enet.fit(train_x, train_y)
	pred_y = enet.predict(test_x)
	np.savetxt(os.path.join(datapath, 'enet_pred.txt'), pred_y)


if __name__ == '__main__':
	if len(sys.argv) < 2:
		print("Usage: svr.py datapath")
		exit()
	else:
		enet(sys.argv[1])
