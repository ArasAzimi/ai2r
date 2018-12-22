from predict_pretrained import pretrained
from keras.models import load_model
import pickle
import numpy as np
import argparse
import json
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,	help="path to input image")
args = vars(ap.parse_args())

image_original = cv2.imread(args["image"])

out_dir = 'out/aircrafts'

# Reading configurations from config.json
with open('config.json') as json_config_file:
	CONFIG = json.load(json_config_file)

# Retrive configuration paramters from the json file
img_w = CONFIG['valid']['img_w']
img_h = CONFIG['valid']['img_h']
img_size = img_w, img_h

# check if any model exists?
print("-- Available trained models --")
# List the sub-dicrectories in out directory:
files = os.listdir(out_dir)
for name, index in enumerate(files):
    print(str(name)+": "+index)
user_model_choice = input("Choose a model for prediction (i.e., 0) > ")
type(user_model_choice)

model = files[int(user_model_choice)].split('_e')[0]

model_file = out_dir+'/'+files[int(user_model_choice)]+'/aircrafts.model'
label_file = out_dir+'/'+files[int(user_model_choice)]+'/aircrafts_lbls.pickle'

# Check if model and label files exists

if model == 'vgg16_pretrained':
	# Setting this to True will run vg116 trained on imagenet first to make sure there is an airplane in the image not a horse!
	run_vgg16_keras_first = True

	if run_vgg16_keras_first == True:
		labels = pretrained.predict_vgg16_keras_imagenet(args["image"])

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
			print(">ia> Loading model and label binarizer...")
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
	print(">ia> Check if model is a correct one!")


(B, G, R) = (0,0,0)
if percent<85: # print in Red
	(B, G, R) = (0,0,255)
elif percent>95: # print in Green
	(B, G, R) = (0,255,0)
else: # print in Yellow
	(B, G, R) =(0,255,255)


legend = "Detected: {}".format(label)
cv2.putText(image_original, legend, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7,	(B, G, R), 2)
cv2.imshow("Image", image_original)
cv2.waitKey(0)
