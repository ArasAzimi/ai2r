from src.predict import Predict

def main():
	import cv2
	import argparse
	import os


	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True,	help="path to input image")
	args = vars(ap.parse_args())
	image = args["image"]
	out_dir = 'out/aircrafts'

	# check if any model exists?
	# Check if model and label files exists
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
	config_file = 'config.json'

	obj = Predict(image, model, model_file, label_file, config_file=config_file)
	res = obj.prediction()

	label = res['label']
	percent = res['percent']

	(B, G, R) = (0,0,0)
	if percent<85: # print in Red
		(B, G, R) = (0,0,255)
	elif percent>95: # print in Green
		(B, G, R) = (0,255,0)
	else: # print in Yellow
		(B, G, R) =(0,255,255)


	legend = "Detected: {}".format(label)
	image_original = cv2.imread(image)
	cv2.putText(image_original, legend, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7,	(B, G, R), 2)
	cv2.imshow("Image", image_original)
	cv2.waitKey(0)

if __name__ == "__main__":
	main()
