import numpy as np
import struct as st
class ann:
	train_labels = 0
	train_images = 0
	test_images = 0
	test_labels = 0
	m_train = 0
	m_test = 0
	def __init__(self, trainloc):
		filename = {'images' : trainloc + 'train-images.idx3-ubyte' ,'labels' : trainloc + 'train-labels.idx1-ubyte'}
		train_imagesfile = open(filename['images'],'rb')
		train_imagesfile.seek(0)
		magic = st.unpack('>4B',train_imagesfile.read(4))
		nImg = st.unpack('>I',train_imagesfile.read(4))[0] 
		nR = st.unpack('>I',train_imagesfile.read(4))[0] 
		nC = st.unpack('>I',train_imagesfile.read(4))[0] 
		nBytesTotal = nImg*nR*nC*1
		self.train_images = np.asarray(st.unpack('>'+'B'*nBytesTotal,train_imagesfile.read(nBytesTotal))).reshape((nImg,nR,nC))
		train_labelsfile = open(filename['labels'],'rb')
		train_labelsfile.seek(0)
		magic = st.unpack('>4B',train_labelsfile.read(4))
		nLabels = st.unpack('>I',train_labelsfile.read(4))[0]
		nBytesTotal = nLabels*1
		self.train_labels = np.asarray(st.unpack('>' + 'B'*nLabels, train_labelsfile.read(nLabels))).reshape((nLabels,1))
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
	
	def activationfn(self, z):
		return 1/(1+np.exp(-z))

	def deriv(self,z):
		return z*(1-z)

	def train(self, X, y ,lam):
		
if __name__ == "__main__":
	trainDatasetLoc = '/home/kauzi/Documents/EveryMLAlgoNumpy/MNIST/Train/'
	#testDatasetLoc = '/home/kauzi/Documents/EveryMLAlgoNumpy/MNIST/Test/'
	obj = ann(trainDatasetLoc)
