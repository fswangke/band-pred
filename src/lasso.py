import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from sklearn.linear_model import Lasso

# load mat
data_numpy = sio.loadmat('data_numpy.mat')
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

# fit lasso should use non-normalized values
alpha = 1
lasso = Lasso(alpha = alpha, max_iter = 10000, tol = 0.5)
lasso.fit(train_x, train_y)
print lasso
# print lasso.coef_     # should plot coef_
# print lasso.score(test_x, test_y) 
pred_y = lasso.predict(test_x)
# print test_y
# print pred_y

# plot results
max_id = 1000
x = list(xrange(len(test_y[1:max_id])))
h_truth = plt.plot(x, test_y[1:max_id], color = 'green',label='Ground truth')
h_pred  = plt.plot(x, pred_y[1:max_id], color = 'red', label='Lasso')
h_base  = plt.plot(x, base_y[1:max_id], color = 'blue', label='[1]')
#plt.legend(handles = [h_truth, h_pred, h_base])
plt.show()
