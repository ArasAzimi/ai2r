class pretrained:
    @staticmethod
    def predict_vgg16_keras_imagenet(in_path):
        # https://machinelearningmastery.com/use-pre-trained-vgg-model-classify-objects-photographs/
        from keras.preprocessing.image import load_img, img_to_array
        from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
        from keras import backend as K
        import numpy as np

        model = VGG16()

        # Load target image
        image = load_img(in_path, target_size=(224, 224))
        # Convert the image pixels to a numpy array
        image = img_to_array(image)
        # Reshape data for the model
        image = np.expand_dims(image, axis=0)
        #image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # Preprocess the image (based on VGG paper), using keras tool
        image = preprocess_input(image)
        # Predict the probability across all output classes
        prediction = model.predict(image)
        # Convert the probabilities to class labels
        labels = decode_predictions(prediction)

        # To release GPU memory after the inception is done
        K.clear_session()
        return labels

    def predict_inceptionv3_keras_imagenet(in_path):
        from keras.preprocessing.image import load_img, img_to_array
        from keras.applications.inception_v3 import InceptionV3, decode_predictions
        from keras import backend as K
        import numpy as np

        model = InceptionV3()

        # Load target image
        image = load_img(in_path, target_size=(224, 224))
        # Convert the image pixels to a numpy array
        image = img_to_array(image)
        # Reshape data for the model
        image = np.expand_dims(image, axis=0)
        #image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # Predict the probability across all output classes
        prediction = model.predict(image)
        # Convert the probabilities to class labels
        labels = decode_predictions(prediction)

        # To release GPU memory after the inception is done
        K.clear_session()
        return labels
