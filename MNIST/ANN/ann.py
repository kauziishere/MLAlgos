"""
	Training MNIST dataset with Artificial neural network with numpy
	2 layered neural network architecture:
	Layers		No. of perceptrons
	Input layer:	784
	Hidden layer:	300
	Output layer:	10
"""
import numpy as np
import struct as st
import random
from tempfile import TemporaryFile
class ann:
	train_labels = 0
	train_images = 0
	mtrain = 60000
	test_labels = 0
	test_images = 0
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
		self.train_images = 255 - np.asarray(st.unpack('>'+'B'*nBytesTotal,train_imagesfile.read(nBytesTotal))).reshape((mtrain,nR,nC))
		print self.train_images[0:1,:]
		train_labelsfile = open(filename['labels'],'rb')
		train_labelsfile.seek(0)
		magic = st.unpack('>4B',train_labelsfile.read(4))
		nLabels = st.unpack('>I',train_labelsfile.read(4))[0]
		nBytesTotal = mtrain*1
		self.train_labels = np.asarray(st.unpack('>' + 'B'*nBytesTotal, train_labelsfile.read(nBytesTotal))).reshape((mtrain,1))

	def activation_fn(self, z):
		return 1/(1+np.exp(-z))

	def deriv(self,z):
		return np.multiply(z, 1-z)

	def variable_initializer(self):
		m = self.mtrain
		X = np.ones((m,self.train_images[0].shape[0]*self.train_images[0].shape[1] + 1))
		X[:,1:] = self.train_images.reshape((m,self.train_images[0].shape[0]*self.train_images[0].shape[1]))
		y = np.zeros(shape=(m,10))
		for i in range(0,m):
			temp = self.train_labels[i][0]
			y[i][temp] = 1
		np.random.seed(0)
		theta1 = np.random.random((60,X.shape[1]))
		theta2 = np.random.random((y.shape[1],theta1.shape[0]+1))
		self.train_images = X
		self.train_labels = y
		self.mtrain = m
		return (theta1, theta2)

	def extract_params(self, file_name_1, file_name_2):
		theta1 = np.load(file_name_1 + '.npy')
		theta2 = np.load(file_name_2 + '.npy')
		return theta1, theta2

	def forward_prop(self, X, theta1, theta2):
		a1 = X
		temp = a1.dot(theta1.T)
		z1 = np.ones((temp.shape[0],temp.shape[1]+1))
		z1[:,1:] = temp
		a2 = self.activation_fn(z1)
		z2 = a2.dot(theta2.T)
		h = self.activation_fn(z2)
		return h

	def cost_function(self, X, y, theta1, theta2):
		m = X.shape[0]
		J = 0
		h = self.forward_prop(X, theta1, theta2)
		first_term = np.sum(np.sum(np.multiply(-y, np.log(h)), axis = 1), axis = 0)
		second_term = np.sum(np.sum(np.multiply(1 - y , np.log(1 - h)), axis = 1), axis = 0)
		J = (1/float(m))*(first_term - second_term)
		return J

	def train(self, theta1 , theta2, lam = 0.03, iterations = 50000):
		X = self.train_images
		y = self.train_labels
		m = self.mtrain
		np.set_printoptions(formatter = {'float_kind':'{:25f}'.format})
		temp1 = 0
		temp2 = 0
	#	print theta1, theta2
		min_j = 999999
		stab_th_1 = 0
		stab_th_2 = 0
		for i in range(0, iterations):
			# Forward propagation
			a1 = X
			temp = a1.dot(theta1.T)
			z1 = np.ones((temp.shape[0],temp.shape[1]+1))
			z1[:,1:] = temp
			a2 = self.activation_fn(z1)
			z2 = a2.dot(theta2.T)
			a3 = self.activation_fn(z2)

			#Back propagation getting delta
			error_2 = a3 - y
			error_1 = np.multiply(error_2.dot(theta2),self.deriv(a2))
			error_1 = error_1[:,1:]
			delta_2 = error_2.T.dot(a2)
			delta_1 = error_1.T.dot(a1)
			
			theta1 -= delta_1/float(m)
			theta2 -= delta_2/float(m)

			if i%100 == 0:
				J = self.cost_function(X, y, theta1, theta2)
				min_j = min(min_j, J)
				if min_j == J:
					name = 'Para_'+str(i//100+1) +'_1'
					name2 = 'Para_'+str(i//100+1) + '_2'
					np.save(name, theta1.astype(np.float))
					np.save(name2, theta2.astype(np.float))
					stab_th_1 = theta1
					stab_th_2 = theta2
				print i/100 + 1, J, min_j

		return stab_th_1, stab_th_2

	def preptest(self, test_set_loc, test_for = 100):
		filename = {'images' : test_set_loc + 't10k-images.idx3-ubyte' ,'labels' : test_set_loc + 't10k-labels.idx1-ubyte'}
		test_imagesfile = open(filename['images'],'rb')
		test_imagesfile.seek(0)
		magic = st.unpack('>4B',test_imagesfile.read(4))
		nImg = st.unpack('>I',test_imagesfile.read(4))[0] 
		nR = st.unpack('>I',test_imagesfile.read(4))[0] 
		nC = st.unpack('>I',test_imagesfile.read(4))[0] 
		nBytesTotal = test_for*nR*nC*1 
		self.test_images =  255 - np.asarray(st.unpack('>'+'B'*nBytesTotal,test_imagesfile.read(nBytesTotal))).reshape((test_for,nR,nC))
		test_labelsfile = open(filename['labels'],'rb')
		test_labelsfile.seek(0)
		magic = st.unpack('>4B',test_labelsfile.read(4))
		nLabels = st.unpack('>I',test_labelsfile.read(4))[0]	
		nBytesTotal = test_for*1
		self.test_labels = np.asarray(st.unpack('>' + 'B'*test_for, test_labelsfile.read(nBytesTotal))).reshape((test_for,1))

	def initialize_test(self, m):
		X = np.ones((m,self.test_images[0].shape[0]*self.test_images[0].shape[1] + 1))
		X[:,1:] = self.test_images.reshape((m,self.test_images[0].shape[0]*self.test_images[0].shape[1]))
		self.test_images = X

	def test(self, test_set_loc, theta1, theta2, test_for = 100):
		self.preptest(test_set_loc, test_for)
		self.initialize_test(test_for)
		#print self.test_images.shape
		print self.test_labels[0:1]
		y = self.forward_prop(self.test_images[0:1], theta1, theta2)
		for i in y:
			for j in range(0,len(i)):
				i[j] = round(i[j])
		print y

if __name__ == "__main__":
	np.set_printoptions(suppress=True)
	trainDatasetLoc = '/home/kauzi/Documents/EveryMLAlgoNumpy/MNIST/Train/'
	testDatasetLoc = '/home/kauzi/Documents/EveryMLAlgoNumpy/MNIST/Test/'
	obj = ann(trainloc = trainDatasetLoc, mtrain = 25000)
	theta1, theta2 = obj.variable_initializer()
	obj.train(theta1, theta2, lam = 0.03, iterations = 15000)
#	theta1, theta2 = obj.extract_params('Para1_119', 'Para2_119')
#	obj.test(testDatasetLoc, theta1, theta2, 100)
