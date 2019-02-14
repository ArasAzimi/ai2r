class Predict():
    """
    Predict class:
    To run prediction and also used for flask application
    """
    def __init__(self, *args, **kwargs):
        self.image = args[0]
        self.model = args[1]
        self.model_file = args[2]
        self.label_file = args[3]
        self.config_file = kwargs['config_file']

    def prediction(self):
    	from predict_pretrained import pretrained
    	from keras.models import load_model
        from keras import backend
    	import pickle
    	import numpy as np
    	import json
    	import cv2

    	image = self.image
    	model = self.model
    	model_file = self.model_file
    	label_file = self.label_file


    	image_original = cv2.imread(image)
    	# Reading configurations from config.json
    	with open(self.config_file) as json_config_file:
    		CONFIG = json.load(json_config_file)

    	# Retrive configuration paramters from the json file
    	img_w = CONFIG['valid']['img_w']
    	img_h = CONFIG['valid']['img_h']
    	img_size = img_w, img_h

    	if model == 'vgg16_pretrained':
    		# Setting this to True will run vg116 trained on imagenet first to make sure there is an airplane in the image not a horse!
    		run_vgg16_keras_first = False

    		if run_vgg16_keras_first == True:
    			labels = pretrained.predict_vgg16_keras_imagenet(image)

    			for i in range(len(labels[0])):
    			    print('%s (%.2f%%)' % (labels[0][i][1], labels[0][i][2]*100))

    			label1 = labels[0][0][1]
    			percent1 = labels[0][0][2]

    			label2 = labels[0][1][1]
    			percent2 = labels[0][1][2]
    			if	label1=='airliner' or label2=='airliner':
    				image = cv2.resize(image_original, (img_w, img_h))
    				image = image.reshape((1, image.shape[0], image.shape[1],image.shape[2]))

    				# Load the model and label binarizer
    				print(">ia> Loading vgg16_pretrained model and label binarizer...")
    				model = load_model(model_file)
    				lb = pickle.loads(open(label_file, "rb").read())
    				predictions = model.predict(image)

    				# Find the class label with the largest probability
    				i = predictions.argmax(axis=1)[0]
    				labels = lb.classes_[i]

    				label = labels
    				percent = predictions[0][i] * 100
    			else:
    				print('>ia> This is not an airliner. Do not run ai2r aircraft type recognition!')
    		else:
    			image = cv2.resize(image_original, (img_w, img_h))
    			image = image.reshape((1, image.shape[0], image.shape[1],image.shape[2]))

    			# Load the model and label binarizer
    			print(">ia> Loading vgg16_pretrained model and label binarizer...")
    			model = load_model(model_file)
    			lb = pickle.loads(open(label_file, "rb").read())
    			predictions = model.predict(image)

    			# Find the class label with the largest probability
    			i = predictions.argmax(axis=1)[0]
    			labels = lb.classes_[i]

    			label = labels
    			percent = predictions[0][i] * 100
    	elif  model == 'inceptionv3_pretrained':
    		image = cv2.resize(image_original, (img_w, img_h))
    		image = image.reshape((1, image.shape[0], image.shape[1],image.shape[2]))

    		# Load the model and label binarizer
    		print(">ia> Loading inceptionv3_pretrained model and label binarizer...")
    		model = load_model(model_file)
    		lb = pickle.loads(open(label_file, "rb").read())
    		predictions = model.predict(image)

    		# Find the class label with the largest probability
    		i = predictions.argmax(axis=1)[0]
    		labels = lb.classes_[i]

    		label = labels
    		percent = predictions[0][i] * 100
    	else:
    		print(">ia> Check if model is a correct one!")

    	res = {"label":label,
               "percent":percent}

        backend.clear_session()
    	return res
