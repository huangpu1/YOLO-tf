import tensorflow as tf
import numpy as np

from nnet import modules as model
import options as opt

class YOLO_tiny(object):

	def __init__(self):
		"""
		defines the architecture of the model
		"""
		self.options = opt.Options()
		self.alpha = self.options.alpha

		# Input to the model
		self.x = tf.placeholder(tf.float32, shape=[None, 448, 448, 3])

		# Stack the layers of the network
		print "Stacking layers of the network\n"
		self.conv_01 = model.conv2d(1, self.x, kernel=[3,3,3,16], stride=2, name='conv_01', alpha=self.alpha)
		self.pool_02 = model.max_pool(2, self.conv_01, name='pool_02')

		self.conv_03 = model.conv2d(3, self.pool_02, kernel=[3,3,16,32], stride=1, name='conv_03', alpha=self.alpha)
		self.pool_04 = model.max_pool(4, self.conv_03, name='pool_04')

		self.conv_05 = model.conv2d(5, self.pool_04, kernel=[3,3,32,64], stride=1, name='conv_05', alpha=self.alpha)
		self.pool_06 = model.max_pool(6, self.conv_05, name='pool_06')

		self.conv_07 = model.conv2d(7, self.pool_06, kernel=[3,3,64,128], stride=1, name='conv_07', alpha=self.alpha)
		self.pool_08 = model.max_pool(8, self.conv_07, name='pool_08')

		self.conv_09 = model.conv2d(9, self.pool_08, kernel=[3,3,128,256], stride=1, name='conv_09', alpha=self.alpha)
		self.pool_10 = model.max_pool(10, self.conv_09, name='pool_10')

		self.conv_11 = model.conv2d(11, self.pool_10, kernel=[3,3,256,512], stride=1, name='conv_11', alpha=self.alpha)
		self.pool_12 = model.max_pool(12, self.conv_11, name='pool_12')

		self.conv_13 = model.conv2d(13, self.pool_12, kernel=[3,3,512,1024], stride=1, name='conv_13', alpha=self.alpha)
		self.conv_14 = model.conv2d(14, self.conv_13, kernel=[3,3,1024,1024], stride=1, name='conv_14', alpha=self.alpha)
		self.conv_15 = model.conv2d(15, self.conv_14, kernel=[3,3,1024,1024], stride=1, name='conv_15', alpha=self.alpha)		

		# Reshape 'self.conv_15' from 4D to 2D
		shape = self.conv_15.get_shape().as_list()
		flat_shape = shape[1] * shape[2] * shape[3]
		inputs_transposed = tf.transpose(self.conv_15, (0,3,1,2))
		fully_flat = tf.reshape(inputs_transposed, [-1, flat_shape])

		self.fc_16 = model.fully_connected(16, fully_flat, 256, name='fc_16', alpha=self.alpha, activation=tf.nn.relu)
		self.fc_17 = model.fully_connected(17, self.fc_16, 4096, name='fc_17', alpha=self.alpha, activation=tf.nn.relu)
		# skip the dropout layer
		self.fc_19 = model.fully_connected(19, self.fc_17, 1470, name='fc_19', alpha=self.alpha, activation=None)
 		
 		self.init_operation = tf.global_variables_initializer()
 		self.saver = tf.train.Saver()


	def count_params(self):
		"""
		Returns the total number of parameters of the model
		"""
		total_parameters = 0
		for variable in tf.trainable_variables():
			count = 1
			for dimension in variable.get_shape().as_list():
				count *= dimension
			total_parameters += count

		return total_parameters


	def train(self):
		"""
		train the model
		"""
		pass

	def validate(self):
		"""
		validate the model
		"""
		pass


	def test(self, test_image):
		"""
		test the model
		"""
		pass
