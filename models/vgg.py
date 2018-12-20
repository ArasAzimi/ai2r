class VGG16_pt:
	#VGG16_pt , 'pt' stands for pretrained
	'''
	Very Deep Convolutional Networks for Large-Scale Image Recognition
	K. Simonyan, A. Zisserman
	arXiv:1409.1556
	'''
	@staticmethod
	def build(width, height, depth, classes):
		"""
		This fucntion will take the pretrained VGG16 model from keras and modify
		the input size, number of classes, or both based on user input.
		"""
		from keras.applications.vgg16 import VGG16
		from keras.preprocessing import image
		from keras.applications.vgg16 import preprocess_input
		from keras.layers import Input, Flatten, Dense
		from keras.models import Model
		import numpy as np

		from keras import backend as K
		img_dim_ordering = 'tf'
		K.set_image_dim_ordering(img_dim_ordering)

		inputShape = (height, width, depth)
		# Get the convolutional part of a VGG network trained on ImageNet
		model = VGG16(weights='imagenet', include_top=False)
		print(">ia> Original pretrained model from keras:\n")
		model.summary()

		# Create your own input format
		input = Input(shape=inputShape,name = 'image_input')

		# Use the generated model
		output_vgg16_conv = model(input)

		#Add the fully-connected layers
		x = Flatten(name='flatten')(output_vgg16_conv)
		x = Dense(4096, activation='relu', name='fc1')(x)
		x = Dense(4096, activation='relu', name='fc2')(x)
		x = Dense(classes, activation='softmax', name='predictions')(x)

		# Create your own model
		model = Model(input=input, output=x)

		#In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
		print(">ia> Modified model using pretrained VGG16 model from keras:\n")
		model.summary()
		return model
