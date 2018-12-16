import numpy as np

def split_dataset(test_size, data, labels):
    from sklearn.model_selection import train_test_split
    """
    partition the data into training and testing splits using test_size.
    test_size % of data is used for cross validation.
    """
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=test_size, random_state=18)
    return trainX, testX, trainY, testY

def prepare_input(imagePaths, img_size):
    """
    Prepare input
    IA: not for production
    """
    import os
    import cv2
    import random

    GRAY_SCALE_VEC = False
    if GRAY_SCALE_VEC:
        print(">ia> Converting to grayscale ...")

    # initialize the data and labels
    print(">ia> Loading images...")
    data = [] # Empty array to hold the input data
    labels = [] # Empty array to hold the input data labels

    random.seed(66)
    random.shuffle(imagePaths)

    # loop over the input images
    for imagePath in imagePaths:
    	# load the image, resize it to 64x64 pixels (the required input
    	# spatial dimensions of SmallVGGNet), and store the image in the
    	# data list
    	image = cv2.imread(imagePath)
    	image = cv2.resize(image, (img_size[0], img_size[1]))
    	#print(image.shape)
    	if GRAY_SCALE_VEC:
    		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    		image = np.reshape(image, (img_size[0]* img_size[1]))
    	#print(image.shape)
    	data.append(image)

    	# extract the class label from the image path and update the
    	# labels list
    	label = imagePath.split(os.path.sep)[-2]
    	labels.append(label)
    print(">ia> Returning `data` and `labels`...")
    return  data, labels

def convert_np(data, labels):
    data = np.array(data, dtype="float")
    data = np.array(data, dtype="float")
    return data, labels

def normalize_input(data):
    # scale the raw pixel intensities to the range [0, 1]
    data = data / 255.0
    return data
