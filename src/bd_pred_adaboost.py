# load data
from sklearn.ensemble import AdaBoostRegressor
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import sys
import os


def adaboost(datapath):
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
	train_y = train_y.ravel()

	# fit lasso should use non-normalized values
	params = {'n_estimators': 500}
	abr = AdaBoostRegressor(**params)
	abr.fit(train_x, train_y)
	pred_y = abr.predict(test_x)
	np.savetxt(os.path.join(datapath, 'adaboost_pred.txt'), pred_y)


if __name__ == '__main__':
	if len(sys.argv) < 2:
		print("Usage: adaboost.py datapath")
		exit()
	else:
		adaboost(sys.argv[1])
