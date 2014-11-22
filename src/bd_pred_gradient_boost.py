# load data
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import sys
import os


def gradient_boost_regressor(datapath):
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

	# visualize one data
	# index = 10
	# x = list(xrange(len(train_x[index])))
	# y = [train_y[index][0] for i in x]
	# plt.plot(x, train_x[index])
	# plt.plot(x, y)
	# plt.show()

	# fit lasso should use non-normalized values
	params = {'n_estimators': 500,
	   'max_depth': 4,
	   'min_samples_split': 1,
	   'learning_rate': 0.01,
	   'loss': 'ls'}
	gbr = GradientBoostingRegressor(**params)
	gbr.fit(train_x, train_y)
	pred_y = gbr.predict(test_x)
	np.savetxt(os.path.join(datapath, 'gradient_boost_pred.txt'), pred_y)


if __name__ == '__main__':
	if len(sys.argv) < 2:
		print("Usage: lasso.py datapath")
		exit()
	else:
		gradient_boost_regressor(sys.argv[1])
