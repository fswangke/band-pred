from sklearn.linear_model import ElasticNet
import numpy as np
import scipy.io as sio
import os
import sys
import time


def elastic_net(datapath):
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

	t_start = time.perf_counter()
	x_fft = np.fft.fft(train_x_raw)
	raw_fft_time = time.perf_counter() - t_start
	train_x_raw_fft = np.concatenate((np.imag(x_fft), np.real(x_fft)), axis=1)
	x_fft = np.fft.fft(test_x_raw)
	test_x_raw_fft = np.concatenate((np.imag(x_fft), np.real(x_fft)), axis=1)

	t_start = time.perf_counter()
	x_fft = np.fft.fft(train_x_smooth)
	smooth_fft_time = time.perf_counter() - t_start
	train_x_smooth_fft = np.concatenate((np.imag(x_fft), np.real(x_fft)), axis=1)
	x_fft = np.fft.fft(test_x_smooth)
	test_x_smooth_fft = np.concatenate((np.imag(x_fft), np.real(x_fft)), axis=1)

	enet_raw = ElasticNet(alpha=0.5, max_iter=50000, tol=0.3)
	t_start = time.perf_counter()
	enet_raw.fit(train_x_raw, train_y)
	elastic_net_raw_time = time.perf_counter() - t_start
	pred_y = enet_raw.predict(test_x_raw)
	np.savetxt(os.path.join(datapath, 'elastic_net_raw.txt'), pred_y)

	enet_raw_fft = ElasticNet(alpha=0.5, max_iter=50000, tol=0.3)
	t_start = time.perf_counter()
	enet_raw_fft.fit(train_x_raw_fft, train_y)
	elastic_net_raw_fft_time = time.perf_counter() - t_start
	pred_y = enet_raw_fft.predict(test_x_raw_fft)
	np.savetxt(os.path.join(datapath, 'elastic_net_raw_fft.txt'), pred_y)

	enet_smooth = ElasticNet(alpha=0.5, max_iter=50000, tol=0.3)
	t_start = time.perf_counter()
	enet_smooth.fit(train_x_smooth, train_y)
	elastic_net_smooth_time = time.perf_counter() - t_start
	pred_y = enet_smooth.predict(test_x_smooth)
	np.savetxt(os.path.join(datapath, 'elastic_net_smooth.txt'), pred_y)

	enet_smooth_fft = ElasticNet(alpha=0.5, max_iter=50000, tol=0.3)
	t_start = time.perf_counter()
	enet_smooth_fft.fit(train_x_smooth_fft, train_y)
	elastic_net_smooth_fft_time = time.perf_counter() - t_start
	pred_y = enet_smooth_fft.predict(test_x_smooth_fft)
	np.savetxt(os.path.join(datapath, 'elastic_net_smooth_fft.txt'), pred_y)

	f_time = open(os.path.join(datapath, 'elastic_net_time.txt'), 'w')
	f_time.write(str(raw_fft_time) + '\n')
	f_time.write(str(smooth_fft_time)+ '\n')
	f_time.write(str(elastic_net_raw_time)+ '\n')
	f_time.write(str(elastic_net_raw_fft_time)+ '\n')
	f_time.write(str(elastic_net_smooth_time)+ '\n')
	f_time.write(str(elastic_net_smooth_fft_time)+ '\n')
	f_time.close()

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print("Usage: svr.py datapath")
		exit()
	else:
		elastic_net(sys.argv[1])
