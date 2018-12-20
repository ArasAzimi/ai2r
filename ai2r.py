from werkzeug.security import safe_str_cmp
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from imutils import paths
import argparse
import json
import os


from util import check, hw_config
from util.prep import *
from util.postp import *

def vgg16_pretrained(img_size,lb):
	"""
	This fucntion will take the pretrained VGG16 model from Keras and modify
	the input size, number of classes, or both based on user input.
	"""
	from models.vgg import VGG16_pt
	print(">ia> Building pretrained vgg16 model...")
	model = VGG16_pt.build(width=img_size[0], height=img_size[1], depth=3, classes=len(lb.classes_))
	return model

def inceptionv3_pretrained(img_size, lb):
	"""
	This function will take the pretrained InceptionV3 model from Keras and modify
	the input size, number of classes, or both based on user input.
	"""
	from models.inception import InceptionV3_pt
	print(">ia> Building pretrained inceptionV3 model...")
	model = InceptionV3_pt.build(width=img_size[0], height=img_size[1], depth=3, classes=len(lb.classes_))
	return model

def select_model(argument, img_size, lb):
	"""
	To select a model based on user input through argument parser.
	"""
	if safe_str_cmp(argument , "inceptionv3_pretrained"):
		model = inceptionv3_pretrained(img_size,lb)
	elif safe_str_cmp(argument , "vgg16_pretrained"):
		model = vgg16_pretrained(img_size, lb)
	else:
		return None

	return model

def str2bool(v):
    """
    Converts various formats of input argument strings for logical true and false
    to python True and False logic.
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
	# Construct argument parser and parse arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--dataset", required=True,	help="path to input dataset of images")
	ap.add_argument("-m", "--model", required=True,	help="name of the model to be used for training")
	ap.add_argument("--gpu", "--gpu", required=True,	help="name of generated files")
	args = vars(ap.parse_args())

	dataset = args["dataset"]
	model_name = args["model"]
	datasets_dir = 'datasets'
	raw_datasets_dir = 'raw'
	RUN_GPU = str2bool(args["gpu"])
	use_raw_data = False

	# Reading configurations from config.json
	with open('config.json') as json_config_file:
		CONFIG = json.load(json_config_file)

	# Retrive configuration paramters from the json file
	GPU_ALLOCATION = 	CONFIG['train']['hw_resource']['USE_GPU']
	test_size = 		CONFIG['train']['test_size']
	img_w = 			CONFIG['train']['img_w']
	img_h = 			CONFIG['train']['img_h']
	learning_rate =	 	CONFIG['train']['learning_rate']
	epochs = 			CONFIG['train']['epochs']
	batch_size = 		CONFIG['train']['batch_size']

	downlaod_dataset_if_not_exists = CONFIG['dataset']['downlaod_ai2r_dataset']
	if downlaod_dataset_if_not_exists:
		use_raw_data = False


	img_size = img_w, img_h
	hw_config.configure_gpu_cpu(RUN_GPU, GPU_ALLOCATION)

	out_dir = 'out/'+ dataset+ '/'+ model_name+ '_e'+ str(epochs)+ '_lr'+ str(learning_rate)+ '_bs'+ str(batch_size)+ '/'
	model_path = out_dir+ dataset
	checkpoint_path=out_dir+ "/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
	np_dataset = datasets_dir+'/'+dataset

	r = check.input_requrements(dataset, np_dataset, downlaod_dataset_if_not_exists)
	if not safe_str_cmp(r , "OK"):
		print(">ia> Exited with error: {}".format(r))
		return 0

	if downlaod_dataset_if_not_exists:
		if os.path.isfile(datasets_dir+'/aircrafts.npy')== False:
			if os.path.isfile('datasets.zip')== False:
				downlaod_dataset(datasets_dir)
			extract_dataset(datasets_dir)

	r = check.model_requirements(model_name, img_size)
	if not safe_str_cmp(r , "OK"):
		print(">ia> Exited with error: {}".format(r))
		return 0

	if os.path.isdir(out_dir) == False:
		print('>ia> Creating output directory:'+ out_dir)
		os.makedirs(out_dir)

	if os.path.isdir(datasets_dir) == False:
		print(">ia> Creating the datasets directory at: /{}".format(datasets_dir))
		os.makedirs(datasets_dir)

	if os.path.isfile(np_dataset+'.npy')== False:
		#if os.path.isdir(datasets_dir) == False:
		print(">ia> Creating numpy dataset from the input...")
		imagePaths = sorted(list(paths.list_images(raw_datasets_dir+'/'+dataset)))
		data, labels = prepare_input(imagePaths, img_size)
		data, labels = convert_np(data, labels)
		data = normalize_input(data)
		np.save(np_dataset, data)
		np.save(np_dataset+'_lbs', labels)
	else:
		print(">ia> Loading previously processed input...")
		data = np.load(np_dataset+'.npy')
		labels = np.load(np_dataset+'_lbs.npy')


	trainX, testX, trainY, testY = split_dataset(test_size, data, labels)


	lb = LabelBinarizer()
	trainY = lb.fit_transform(trainY)
	testY = lb.transform(testY)

	model = select_model(model_name, img_size, lb)
	if model == None:
		print(">ia> Exited with error: model_error")
		return 0

	# Initialize model and optimizer ( use binary_crossentropy for 2-class classification)
	opt = SGD(lr=learning_rate, decay=learning_rate / epochs)
	model.compile(loss="categorical_crossentropy", optimizer=opt,	metrics=["accuracy"])

	# Train the model
	train_datagen = ImageDataGenerator(
	        rescale=1,
			rotation_range=30,
			width_shift_range=0.1,
			height_shift_range=0.1,
			fill_mode="nearest",
	        shear_range=0.2,
	        zoom_range=0.2,
	        horizontal_flip=True)

	test_datagen = ImageDataGenerator(rescale=1)

	train_generator = train_datagen.flow(
	        trainX,
			trainY,
			batch_size=batch_size)

	validation_generator = test_datagen.flow(
			testX,
			testY,
			batch_size=batch_size)

	model_fit= model.fit_generator(
	        train_generator,
	        steps_per_epoch=len(trainX) // batch_size,
	        epochs=epochs,
	        validation_data=validation_generator,
	        validation_steps=len(testX) // batch_size,
			callbacks=None)

	# Evaluate the network and save the report to eval_report
	print(">ia> Evaluating network...")
	predictions = model.predict(testX, batch_size=batch_size)
	eval_report = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_)


	print(eval_report)
	# plot the training loss and accuracy
	plot_save_results(model_fit, model_path)
	save_model(model, model_path, lb, eval_report, CONFIG)

# Python 3 does not need if __name__ == "__main__":
if __name__ == "__main__":
    main()
