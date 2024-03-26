import numpy as np
import gzip
import os
import platform
import pickle


class DatasetLoad(object):
	def __init__(self, dataSetName, isIID):
		self.name = dataSetName
		self.train_data = None
		self.train_label = None
		self.train_data_size = None
		self.test_data = None
		self.test_label = None
		self.test_data_size = None

		self._index_in_train_epoch = 0

		if self.name == 'femnist':
			self.oarfDataSetConstruct(isIID)
		else:
			pass


	def oarfDataSetConstruct(self, isIID):
		data_dir = 'data/OARF'
		train_data_path = os.path.join(data_dir, 'FEMINIST.gz')
		train_labels_path = os.path.join(data_dir, 'CIFAR-10.gz')
		test_data_path = os.path.join(data_dir, 'Sent140.gz')
		test_labels_path = os.path.join(data_dir, 'Train_and_Test.gz')
		train_data = extract_data(train_data_path)
		train_labels = extract_labels(train_labels_path)
		test_data = extract_data(test_data_path)
		test_labels = extract_labels(test_labels_path)
		# CPU reduce size
		# train_data = train_data[:60]
		# train_labels = train_labels[:60]
		# test_data = test_data[:60]
		# test_labels = test_labels[:60]

		# 60000 data points
		assert train_data.shape[0] == train_labels.shape[0]
		assert test_data.shape[0] == test_labels.shape[0]

		self.train_data_size = train_data.shape[0]
		self.test_data_size = test_data.shape[0]

		assert train_data.shape[3] == 1
		assert test_data.shape[3] == 1
		train_data = train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2])
		test_data = test_data.reshape(test_data.shape[0], test_data.shape[1] * test_data.shape[2])

		train_data = train_data.astype(np.float32)
		train_data = np.multiply(train_data, 1.0 / 255.0)
		test_data = test_data.astype(np.float32)
		test_data = np.multiply(test_data, 1.0 / 255.0)

		if isIID:
			order = np.arange(self.train_data_size)
			np.random.shuffle(order)
			self.train_data = train_data[order]
			self.train_label = train_labels[order]
		else:
			labels = np.argmax(train_labels, axis=1)
			order = np.argsort(labels)
			self.train_data = train_data[order]
			self.train_label = train_labels[order]



		self.test_data = test_data
		self.test_label = test_labels


def _read32(bytestream):
	dt = np.dtype(np.uint32).newbyteorder('>')
	return np.frombuffer(bytestream.read(4), dtype=dt)[0]

database_name = {"FEMINIST.gz": "FEMINIST Dataset",
				"CIFAR-10.gz": "CIFAR-10 Dataset", 
				"Sent140.gz": "Sent140 Dataset", 
				"Train_and_Test.gz": "Train and Test"}

def extract_data(filename):
	"""Extract the data into a 4D uint8 numpy array [index, y, x, depth]."""

	print('Extracting', database_name[filename.split('/')[-1]])
	with gzip.open(filename) as bytestream:
		magic = _read32(bytestream)
		if magic != 2051:
			raise ValueError(
					'Invalid magic number %d in OARF data file: %s' %
					(magic, filename))
		num_data = _read32(bytestream)
		rows = _read32(bytestream)
		cols = _read32(bytestream)
		buf = bytestream.read(rows * cols * num_data)
		data = np.frombuffer(buf, dtype=np.uint8)
		data = data.reshape(num_data, rows, cols, 1)
		return data


def dense_to_one_hot(labels_dense, num_classes=10):
	"""Convert class labels from scalars to one-hot vectors."""
	num_labels = labels_dense.shape[0]
	index_offset = np.arange(num_labels) * num_classes
	labels_one_hot = np.zeros((num_labels, num_classes))
	labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
	return labels_one_hot


def extract_labels(filename):
	"""Extract the labels into a 1D uint8 numpy array [index]."""
	print('Extracting', database_name[filename.split('/')[-1]])
	with gzip.open(filename) as bytestream:
		magic = _read32(bytestream)
		if magic != 2049:
			raise ValueError(
					'Invalid magic number %d in FEMNIST label file: %s' %
					(magic, filename))
		num_items = _read32(bytestream)
		buf = bytestream.read(num_items)
		labels = np.frombuffer(buf, dtype=np.uint8)
		return dense_to_one_hot(labels)


if __name__=="__main__":
	'test data set'
	oarfDataSet = GetDataSet('femnist', True) # test NON-IID
	if type(oarfDataSet.train_data) is np.ndarray and type(oarfDataSet.test_data) is np.ndarray and \
			type(oarfDataSet.train_label) is np.ndarray and type(oarfDataSet.test_label) is np.ndarray:
		print('the type of data is numpy ndarray')
	else:
		print('the type of data is not numpy ndarray')
	print('the shape of the train data set is {}'.format(oarfDataSet.train_data.shape))
	print('the shape of the test data set is {}'.format(oarfDataSet.test_data.shape))
	print(oarfDataSet.train_label[0:100], oarfDataSet.train_label[11000:11100])

# Data Poisoning Attack
# add Gussian Noise to dataset

class AddGaussianNoise(object):
	def __init__(self, mean=0., std=1.):
		self.std = std
		self.mean = mean
		
	def __call__(self, tensor):
		return tensor + torch.randn(tensor.size()) * self.std + self.mean
	
	def __repr__(self):
		return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
