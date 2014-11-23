from sklearn.linear_model import Lasso
import numpy as np
import scipy.io as sio
import sys
import os


def lasso(datapath):
	# load mat
	datafile = os.path.join(datapath, 'data_numpy.mat')
	if os.path.exists(datafile) is False:
		print('Data file %s not found.' % datafile)

	data_numpy = sio.loadmat(datafile)
	# get training and test data
	train_x_raw = data_numpy['trainX_raw'];
	train_x_smooth= data_numpy['trainX_smooth'];
	train_y = data_numpy['trainY'];
	test_x_raw  = data_numpy['testX_raw'];
	test_x_smooth = data_numpy['testX_smooth'];
	test_y  = data_numpy['testY'];
	base_y  = data_numpy['baseY'];

	train_y = train_y.ravel()

	x_fft = np.fft.fft(train_x_raw)
	train_x_raw_fft = np.concatenate((np.imag(x_fft), np.real(x_fft)), axis=1)
	x_fft = np.fft.fft(test_x_raw)
	test_x_raw_fft = np.concatenate((np.imag(x_fft), np.real(x_fft)), axis=1)

	x_fft = np.fft.fft(train_x_smooth)
	train_x_smooth_fft = np.concatenate((np.imag(x_fft), np.real(x_fft)), axis=1)
	x_fft = np.fft.fft(test_x_smooth)
	test_x_smooth_fft = np.concatenate((np.imag(x_fft), np.real(x_fft)), axis=1)

	# fit lasso should use non-normalized values
	alpha = 1
	lasso_raw = Lasso(alpha = alpha, max_iter = 10000, tol = 0.5)
	lasso_raw.fit(train_x_raw, train_y)
	print(lasso_raw)
	pred_y = lasso_raw.predict(test_x_raw)
	np.savetxt(os.path.join(datapath, 'lasso_raw.txt'), pred_y)

	alpha = 1
	lasso_raw_fft = Lasso(alpha = alpha, max_iter = 10000, tol = 0.5)
	lasso_raw_fft.fit(train_x_raw_fft, train_y)
	print(lasso_raw)
	pred_y = lasso_raw_fft.predict(test_x_raw_fft)
	np.savetxt(os.path.join(datapath, 'lasso_raw_fft.txt'), pred_y)

	alpha = 1
	lasso_smooth = Lasso(alpha = alpha, max_iter = 10000, tol = 0.5)
	lasso_smooth.fit(train_x_smooth, train_y)
	print(lasso_smooth)
	pred_y = lasso_smooth.predict(test_x_smooth)
	np.savetxt(os.path.join(datapath, 'lasso_smooth.txt'), pred_y)

	alpha = 1
	lasso_smooth_fft = Lasso(alpha = alpha, max_iter = 10000, tol = 0.5)
	lasso_smooth_fft.fit(train_x_smooth_fft, train_y)
	print(lasso_smooth_fft)
	pred_y = lasso_smooth_fft.predict(test_x_smooth_fft)
	np.savetxt(os.path.join(datapath, 'lasso_smooth_fft.txt'), pred_y)


if __name__ == '__main__':
	if len(sys.argv) < 2:
		print("Usage: lasso.py datapath")
		exit()
	else:
		lasso(sys.argv[1])
