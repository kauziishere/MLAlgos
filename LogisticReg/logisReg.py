import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def sigmoid(z):
	return 1/(1+(math.e)**(-z))
def visualize(X1, X2, y):
	fig=plt.figure()
	ax = Axes3D(fig)
	ax.scatter(xs = X1, ys = X2, zs = y, label = 'curve')
	plt.show()
def train(X, y, theta, learning_rate = 0.01, iterations = 1500, vis = True):
	m = y.shape[0]
	X_0 = np.zeros(shape=(m,1))
	X_0[:,0] = X[:,0]
	X_1 = np.zeros(shape=(m,1))
	X_1[:,0] = X[:,1]
	X_2 = np.zeros(shape=(m,1))
	X_2[:,0] = X[:,2]
	#print X_2
	visualize(X_1, X_2, y)
	for i in xrange(iterations):
		h = sigmoid(X.dot(theta)) - y
		grad0 =  ((learning_rate)*(1.0/m)*((X_0*h).sum(axis=0)))
		grad1 = (learning_rate)*(1.0/m)*((X_1*(h)).sum(axis=0))
		grad2 = (learning_rate)*(1.0/m)*((X_2*(h)).sum(axis=0))
		theta[0][0] -= grad0
		theta[1][0] -= grad1
		theta[2][0] -= grad2
	#print theta
	return theta
if __name__ == "__main__":
	data = pd.read_csv('dataset.csv')
	m = len(data['Y'])
	y = np.zeros(shape=(m,1))
	y[:,0] = data['Y']
	X = np.ones(shape=(m,3))
	X[:,1] = data['X1']
	X[:,2] = data['X2']
	learning_rate = 0.05
	theta = np.ones(shape=(3,1))
	iterations = 1500
	theta = train(X, y, theta, learning_rate, iterations, True)
	x = np.array([1,2.7810836,2.550537003])
	print round(sigmoid(x.dot(theta))[0])
