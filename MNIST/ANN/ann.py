"""
	Training MNIST dataset with Artificial neural network with numpy
	2 layered neural network architecture:
	Layers		No. of perceptrons
	Input layer:	784
	Hidden layer:	30
	Output layer:	10
"""
import numpy as np
import struct as st
import random

class ann:
	train_labels = 0
	train_images = 0
	mtrain = 60000
	test_labels = 0
	test_images = 0
	theta1 = 0
	theta2 = 0
	theta1 = 0
	theta2 = 0
	def __init__(self, trainloc, mtrain = 5000):
		self.mtrain = mtrain
		filename = {'images' : trainloc + 'train-images.idx3-ubyte' ,'labels' : trainloc + 'train-labels.idx1-ubyte'}
		train_imagesfile = open(filename['images'],'rb')
		train_imagesfile.seek(0)
		magic = st.unpack('>4B',train_imagesfile.read(4))
		nImg = st.unpack('>I',train_imagesfile.read(4))[0] 
		nR = st.unpack('>I',train_imagesfile.read(4))[0] 
		nC = st.unpack('>I',train_imagesfile.read(4))[0] 
		nBytesTotal = mtrain*nR*nC*1
		self.train_images = np.asarray(st.unpack('>'+'B'*nBytesTotal,train_imagesfile.read(nBytesTotal))).reshape((mtrain,nR,nC))
		train_labelsfile = open(filename['labels'],'rb')
		train_labelsfile.seek(0)
		magic = st.unpack('>4B',train_labelsfile.read(4))
		nLabels = st.unpack('>I',train_labelsfile.read(4))[0]
		nBytesTotal = mtrain*1
		self.train_labels = np.asarray(st.unpack('>' + 'B'*nBytesTotal, train_labelsfile.read(nBytesTotal))).reshape((mtrain,1))

	def activation_fn(self, z):
		return 1/(1+np.exp(-z))

	def deriv(self,z):
		return z*(1-z)

	def variable_initializer(self):
		m = self.mtrain
		X = self.train_images.reshape((m,self.train_images[0].shape[0]*self.train_images[0].shape[1]))
		y = np.zeros(shape=(m,10))
		for i in range(0,m):
			temp = self.train_labels[i][0]
			y[i][temp] = 1
		theta1 = np.random.random((X.shape[1],30))
		theta2 = np.random.random((theta1.shape[1],y.shape[1]))
		self.train_images = X
		self.train_labels = y
		self.mtrain = m
		return theta1, theta2

	def forward_prop(self, X, theta1, theta2):
		a1 = X
		z1 = a1.dot(theta1)
		a2 = self.activation_fn(z1)
		z2 = a2.dot(theta2)
		a3 = self.activation_fn(z2)
		return a3

	def train(self, theta1 , theta2, lam = 0.03, iterations = 50000):
		X = self.train_images
		y = self.train_labels
		m = self.mtrain
		for i in xrange(iterations):
			# Forward propagation
			a1 = X
			z1 = X.dot(theta1)
			a2 = self.activation_fn(z1)
			z2 = a2.dot(theta2)
			a3 = self.activation_fn(z2)

			#Back propagation getting delta
			error_2 = a3 - y
			delta_2 = error_2*self.deriv(a3)
			error_1 = error_2.dot(theta2.T)
			delta_1 = error_1*self.deriv(a2)

			#updating theta
			theta2 -= a2.T.dot(delta_2)*lam
			theta1 -= a1.T.dot(delta_1)*lam

			if i%100 == 0:
				val = self.forward_prop(X[0],theta1, theta2)
				for i in range(0,len(val)):
					val[i] = round(val[i])
				print i/100,':'
				print val 
				print y[0]

		return theta1, theta2

	def preptest(self, test_set_loc, test_for = 100):
		filename = {'images' : test_set_loc + 't10k-images.idx3-ubyte' ,'labels' : test_set_loc + 't10k-labels.idx1-ubyte'}
		test_imagesfile = open(filename['images'],'rb')
		test_imagesfile.seek(0)
		magic = st.unpack('>4B',test_imagesfile.read(4))
		nImg = st.unpack('>I',test_imagesfile.read(4))[0] 
		nR = st.unpack('>I',test_imagesfile.read(4))[0] 
		nC = st.unpack('>I',test_imagesfile.read(4))[0] 
		nBytesTotal = test_for*nR*nC*1 
		self.test_images = np.asarray(st.unpack('>'+'B'*nBytesTotal,test_imagesfile.read(nBytesTotal))).reshape((test_for,nR,nC))
		test_labelsfile = open(filename['labels'],'rb')
		test_labelsfile.seek(0)
		magic = st.unpack('>4B',test_labelsfile.read(4))
		nLabels = st.unpack('>I',test_labelsfile.read(4))[0]	
		nBytesTotal = test_for*1
		self.test_labels = np.asarray(st.unpack('>' + 'B'*test_for, test_labelsfile.read(nBytesTotal))).reshape((test_for,1))

	def initialize_test(self, m):
		X = self.test_images.reshape((m,self.test_images[0].shape[0]*self.test_images[0].shape[1]))
		self.test_images = X

	def test(self, test_set_loc, theta1, theta2, test_for = 100):
		self.preptest(test_set_loc, test_for)
		self.initialize_test(test_for)
		print self.test_images.shape
		print self.test_labels[0:5]
		print self.forward_prop(self.test_images, theta1, theta2)[0:5]

if __name__ == "__main__":
	trainDatasetLoc = '/home/kauzi/Documents/EveryMLAlgoNumpy/MNIST/Train/'
	testDatasetLoc = '/home/kauzi/Documents/EveryMLAlgoNumpy/MNIST/Test/'
	obj = ann(trainloc = trainDatasetLoc, mtrain = 10000)
	theta1, theta2 = obj.variable_initializer()
	obj.train(theta1, theta2, lam = 0.1, iterations = 1500)
	obj.test(testDatasetLoc, theta1, theta2, 100)
