print(__doc__)

from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os
import sys


def svr(datapath):
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

	# fit svr regression model
	# RBF kernel
	svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
	svr_rbf.fit(train_x, train_y)
	# linear kernel
	svr_lin = SVR(kernel='linear', C=1e3)
	svr_lin.fit(train_x, train_y)
	# polynomial kernel
	svr_poly = SVR(kernel='poly', C=1e3, degree=2)
	svr_poly.fit(train_x, train_y)


if __name__ == '__main__':
	if len(sys.argv) < 2:
		print("Usage: svr.py datapath")
		exit()
	else:
		svr(sys.argv[1])
