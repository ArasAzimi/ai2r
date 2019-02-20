from flask import request, current_app
from flask_restful import Resource, reqparse
import os, sys

sys.path.append(os.path.abspath(os.path.join('..')))

from src.predict import Predict

class PredictResource(Resource):
    """
    Flask resource for prediction
    """
    parser = reqparse.RequestParser()
    parser.add_argument(
                        'filename',
                        type= str,
                        required = True,
                        help="Filename is required!"
    )
    parser.add_argument(
                        'modelname',
                        type= str,
                        required = True,
                        help="Model name is required!"
    )

    def __init__(self):
        data = PredictResource.parser.parse_args()
        filename = data['filename']
        modelname = data['modelname']
        self.filename = filename
        self.modelname = modelname

    def post(self):
        data = self.parser.parse_args()
        try:
            label= PredictModel(data)
        except ValueError:
            return jsonify("Please enter a valid model_name.")

        return label

    def get(self):
        data = request.get_json()
        image = os.path.join(current_app.config['UPLOAD_FOLDER'], self.filename)
        model_file = os.path.join(current_app.config['MODEL_FOLDER'], self.modelname+ '.model')
        label_file = os.path.join(current_app.config['MODEL_FOLDER'], self.modelname+ '_lbls.pickle')
        config_file = current_app.config['MODEL_CONFIG_FILE']
        obj = Predict(image, self.modelname, model_file, label_file, config_file=config_file)
        res = obj.prediction()
        label = res['label']
        #label = predict_api(data)
        return{"label":label}

    def put(self):
        pass

    def delete(self):
        pass

def predict_api(data):
    # imports

    import argparse
    import cv2
    from keras import backend as K
    from keras.models import load_model
    import pickle

    filename=data["filename"]
    model = data["model_name"]
    full_filename = './static/aircrafts/'+filename
    image_original = cv2.imread(full_filename)

    """import os
    if os.path.isfile(full_filename):
        return "good"
    else:
        return full_filename"""

    model_file = model+'.model' #out_dir+'/'+files[int(user_model_choice)]+'/aircrafts.model'
    label_file = model+'_lbls.pickle' #out_dir+'/'+files[int(user_model_choice)]+'/aircrafts_lbls.pickle'

    # Retrive configuration paramters from the json file
    img_w = 139 #CONFIG['valid']['img_w']
    img_h = 139 #CONFIG['valid']['img_h']
    img_size = img_w, img_h

    if model == 'inceptionv3':
        # Setting this to True will run vg116 trained on imagenet first to make sure there is an airplane in the image not a horse!
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
    elif model == "vgg16":
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
    K.clear_session()
    return label
