# Linear regression univariate
# Equation:	 Y = 2*X + 2
import numpy as np
import pandas as pd
def LinearReg(X, y, theta,learning_rate=0.01,iterations=1500):
	X_1 = np.zeros(shape=(m,1))
	X_1.reshape((m,1))
	X_1[:,0] = X[:,1]
	for i in xrange(0,iterations):
		h0 = X.dot(theta) - y
		h1 = (X.dot(theta) - y)*X_1
		grad0 = (learning_rate)*(1.0/m)*(h0)
		grad1 = (learning_rate)*(1.0/m)*(h1)
		theta[0][0] = theta[0][0] - grad0.sum()
		theta[1][0] = theta[1][0] - grad1.sum()
	return theta
def test(theta):
	x = np.ones(shape=(1,2))
	x[0][1] = int(input("Get value"))
	print x, theta
	print x.dot(theta)
if __name__ == "__main__":
	data = pd.read_csv("equation.csv")
	temp = data['Y']
	m = len(temp)
	np.set_printoptions(suppress=True)
	y = np.zeros(shape=(m,1))
	y[:,0] = temp
	X = np.ones(shape=(m,2))
	X[:,1] = data['X']
	theta = np.zeros(shape=(2,1))
	learning_rate=0.01
	iterations = 1500
	theta = LinearReg(X, y, theta, learning_rate, iterations)
	test(theta)
