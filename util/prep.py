import numpy as np

def download_dataset(datasets_dir):
    """
    Downloads data from an online server
    """
    import urllib.request
    print(">ia> Downloading numpy dataset for ai2r...")
    datasets_url = "https://www.dropbox.com/s/tc1gg44u6iuetax/datasets.zip?dl=1"
    u = urllib.request.urlopen(datasets_url)
    data = u.read()
    u.close()
    filename = datasets_dir+'.zip'
    with open(filename, "wb") as f :
        f.write(data)

def extract_dataset(datasets_dir):
    """
    Extracts the downloaded zip data
    """
    import zipfile
    print(">ia> Extracting numpy dataset for ai2r...")
    filename = datasets_dir+'.zip'
    zip_ = zipfile.ZipFile(filename)
    zip_.extractall(datasets_dir)
    zip_.close()


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

    gray_scale_vec = False
    if gray_scale_vec:
        print(">ia> Converting to grayscale ...")

    # initialize the data and labels
    print(">ia> Loading images...")
    data = [] # Empty array to hold the input data
    labels = [] # Empty array to hold the input data labels

    random.seed(66)
    random.shuffle(imagePaths)

    # loop over the input images
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (img_size[0], img_size[1]))

        if gray_scale_vec:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.reshape(image, (img_size[0]* img_size[1]))
        data.append(image)

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
