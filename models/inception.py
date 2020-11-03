# Downloading data from https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
class InceptionV3_pt:
	#InceptionV3_pt , 'pt' stands for pretrained
	@staticmethod
	def build(width, height, depth, classes):
		"""
		This fucntion will take the pretrained InceptionV3 model from keras and modify
		the input size, number of classes, or both based on user input.
		"""
		from keras.applications.inception_v3 import InceptionV3
		from keras.preprocessing import image
		from keras.layers import Input, Flatten, Dense
		from keras.models import Model
		import numpy as np

		from keras import backend as K
		img_dim_ordering = 'tf'
		K.set_image_dim_ordering(img_dim_ordering)

		inputShape = (height, width, depth)
		# model = InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
		model = InceptionV3(include_top=False, weights='imagenet', input_shape=inputShape,)
		#print(">ia> Original pretrained model from keras:\n")
		#model.summary()

		# Create your own input format
		input_image = Input(shape=inputShape,name = 'image_input')

		# Use the generated model
		output_incv3_conv = model(input_image)

		#Add the fully-connected layers
		x = Flatten(name='flatten')(output_incv3_conv)
		x = Dense(4096, activation='relu', name='fc1')(x)
		x = Dense(4096, activation='relu', name='fc2')(x)
		x = Dense(classes, activation='softmax', name='predictions')(x)

		#Create your own model
		model = Model(input=input_image, output=x)

		print(">ia> Modified model using pretrained InceptionV3 model from keras:\n")
		model.summary()
		return model
