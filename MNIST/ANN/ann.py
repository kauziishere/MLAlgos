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
		"""
		filename = {'images' : testloc + 't10k-images.idx3-ubyte' ,'labels' : trainloc + 't10k-labels.idx1-ubyte'}
		test_imagesfile = open(filename['images'],'rb')
		test_imagesfile.seek(1)
		magic = st.unpack('>4B',test_imagesfile.read(4))
		nImg = st.unpack('>I',test_imagesfile.read(4))[0] 
		nR = st.unpack('>I',test_imagesfile.read(4))[0] 
		nC = st.unpack('>I',test_imagesfile.read(4))[0] 
		nBytesTotal = nImg*nR*nC*1 
		self.test_images = np.asarray(st.unpack('>'+'B'*nBytesTotal,test_imagesfile.read(nBytesTotal))).reshape((nImg,nR,nC))
		test_labelsfile = open(filename['labels'],'rb')
		test_labelsfile.seek(1)
		magic = st.unpack('>4B',test_labelsfile.read(4))
		nLabels = st.unpack('>I',test_labelsfile.read(4))[0]
		nBytesTotal = nLabels*1
		self.test_labels = np.asarray(st.unpack('>' + 'B'*nLabels, test_labelsfile.read(nLabels))).reshape((nLabels,1)) """

	def activation_fn(self, z):
		return 1/(1+np.exp(-z))

	def deriv(self,z):
		return z*(1-z)

	def variable_initializer(self, lam = 0.03, iterations = 50000):
		m = self.mtrain
		X = self.train_images.reshape((m,self.train_images[0].shape[0]*self.train_images[0].shape[1]))
		y = np.zeros(shape=(m,10))
		for i in range(0,m):
			temp = self.train_labels[i][0]
			y[i][temp] = 1
		theta1 = np.random.random((X.shape[1],30))
		theta2 = np.random.random((theta1.shape[0],y.shape[1]))
		self.train_images = X
		self.train_labels = y
		self.mtrain = m
		return theta1, theta2
		
if __name__ == "__main__":
	trainDatasetLoc = '/home/kauzi/Documents/EveryMLAlgoNumpy/MNIST/Train/'
	#testDatasetLoc = '/home/kauzi/Documents/EveryMLAlgoNumpy/MNIST/Test/'
	obj = ann(trainloc = trainDatasetLoc, mtrain = 5000)
	theta1, theta2 = obj.variable_initializer(lam = 0.01, iterations = 50000)
