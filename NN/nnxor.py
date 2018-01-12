"""
	Neural network with 1 hidden layer for XOR gate with 2 inputs
	Layer		Perceptrons
	Input layer: 	2
	Hidden layer:	2
	Output layer:	1
"""
import numpy as np
import pandas as pd
class neuralNetwork:
	X = 0
	y = 0
	m = 0
	theta1 = 0
	theta2 = 0
	def __init__(self, loc):
		data = pd.read_csv(loc)
		self.m = len(data['Y'])
		self.y = np.zeros(shape=(self.m,1))
		self.y[:,0] = data['Y']
		self.X = np.ones(shape=(self.m,2))
		self.X[:,0] = data['X1']
		self.X[:,1] = data['X2']
		np.random.seed(0)
		self.theta1 = np.random.random((2,2))
		self.theta2 = np.random.random((2,1))

	# Activation function
	def activationfn(self, z):
		return 1/(1+(np.exp(-z)))

	# Forward propogation
	def forwardProp(self, X, theta1, theta2):
		a1 = X
		z1 = a1.dot(theta1)
		a2 = self.activationfn(z1)
		z2 = a2.dot(theta2)
		a3 = self.activationfn(z2)
		return a3

	# Training the data set to get weights. 
	#Theta1 refers to weights from input layer to hidden layer
	#Theta2 refers to weights from hidden layer to output layer
	#Lam is the learning rate
	def train(self, theta1, theta2, lam = 0.1 , iterations = 50000):
		for i in xrange(iterations):
			a1 = self.X		
			z1 = a1.dot(theta1)
			a2 = self.activationfn(z1)
			z2 = a2.dot(theta2)
			a3 = self.activationfn(z2)
			error2 = a3 - self.y
			delta2 = error2*(a3*(1-a3))
			delta1 = (error2.dot(theta2.T))*(a2*(1-a2))
			theta2 -= a2.T.dot(delta2)*lam
			theta1 -= a1.T.dot(delta1)*lam
			if i%10000 == 0:
				print 'Error at %d th iteration is:' % i , np.mean(np.abs(error2))
		self.theta1 = theta1
		self.theta2 = theta2
	
	# Check custom test case
	def customtest(self, theta1, theta2):
		x1 , x2 = raw_input("Enter custom x1 and x2: ").split()
		X = np.ones(shape = (1,2))
		X[0,:] = [x1,x2]
		print round(self.forwardProp(X,theta1, theta2)) 

if __name__ == "__main__":
	nn = neuralNetwork('XOR.csv')
	nn.train(nn.theta1, nn.theta2, 0.1, 60000)
	nn.customtest(nn.theta1, nn.theta2)
