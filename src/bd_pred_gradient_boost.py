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

	params = {'n_estimators': 200}
	grb_raw = GradientBoostingRegressor(**params)
	grb_raw.fit(train_x_raw, train_y)
	pred_y = grb_raw.predict(test_x_raw)
	np.savetxt(os.path.join(datapath, 'gradient_boost_raw.txt'), pred_y)

	grb_raw_fft = GradientBoostingRegressor(**params)
	grb_raw_fft.fit(train_x_raw_fft, train_y)
	pred_y = grb_raw_fft.predict(test_x_raw_fft)
	np.savetxt(os.path.join(datapath, 'gradient_boost_raw_fft.txt'), pred_y)

	grb_smooth = GradientBoostingRegressor(**params)
	grb_smooth.fit(train_x_smooth, train_y)
	pred_y = grb_smooth.predict(test_x_smooth)
	np.savetxt(os.path.join(datapath, 'gradient_boost_smooth.txt'), pred_y)

	grb_smooth_fft = GradientBoostingRegressor(**params)
	grb_smooth_fft.fit(train_x_smooth_fft, train_y)
	pred_y = grb_smooth_fft.predict(test_x_smooth_fft)
	np.savetxt(os.path.join(datapath, 'gradient_boost_smooth_fft.txt'), pred_y)


if __name__ == '__main__':
	if len(sys.argv) < 2:
		print("Usage: lasso.py datapath")
		exit()
	else:
		gradient_boost_regressor(sys.argv[1])
