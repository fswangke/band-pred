import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io as sio
import sys

from scipy.ndimage import convolve
from sklearn import linear_model
from sklearn import metrics
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

def rbm(datapath):
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

	# models
	logistic = linear_model.LogisticRegression()
	rbms = BernoulliRBM(random_state=0, verbose=True)

	regressor = Pipeline(steps=[('rbm', rbms), ('logistic', logistic)])

	# training
	rbms.learning_rate = 0.05
	rbms.n_iter = 30
	rbms.n_components = 100
	logistic.C = 6000.0

	regressor.fit(train_x, train_y)

	# evaluation
	pred_y = regressor.predict(test_x)
	np.savetxt(os.path.join(datapath, 'rbm_logistic_pred.txt'), pred_y)


if __name__ == '__main__':
	if len(sys.argv) < 2:
		print("Usage: adaboost.py datapath")
		exit()
	else:
		rbm(sys.argv[1])
