from sklearn.ensemble import RandomForestRegressor
import numpy as np
import scipy.io as sio
import sys
import os


def random_forest_regressor(datapath):
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

	rfr_raw = RandomForestRegressor()
	rfr_raw.fit(train_x_raw, train_y)
	pred_y = rfr_raw.predict(test_x_raw)
	np.savetxt(os.path.join(datapath, 'random_forest_raw.txt'), pred_y)

	rfr_raw_fft = RandomForestRegressor()
	rfr_raw_fft.fit(train_x_raw_fft, train_y)
	pred_y = rfr_raw_fft.predict(test_x_raw_fft)
	np.savetxt(os.path.join(datapath, 'random_forest_raw_fft.txt'), pred_y)

	rfr_smooth = RandomForestRegressor()
	rfr_smooth.fit(train_x_smooth, train_y)
	pred_y = rfr_smooth.predict(test_x_smooth)
	np.savetxt(os.path.join(datapath, 'random_forest_smooth.txt'), pred_y)

	rfr_smooth_fft = RandomForestRegressor()
	rfr_smooth_fft.fit(train_x_smooth_fft, train_y)
	pred_y = rfr_smooth_fft.predict(test_x_smooth_fft)
	np.savetxt(os.path.join(datapath, 'random_forest_smooth_fft.txt'), pred_y)


if __name__ == '__main__':
	if len(sys.argv) < 2:
		print("Usage: lasso.py datapath")
		exit()
	else:
		random_forest_regressor(sys.argv[1])
