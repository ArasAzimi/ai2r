class Vgg16_pt:
    """
    Very Deep Convolutional Networks for Large-Scale Image Recognition
    K. Simonyan, A. Zisserman
    arXiv:1409.1556
    # VGG16_pt , 'pt' stands for pretrained
    """

    @staticmethod
    def build(width, height, depth, classes):
        """
        This function will take the pretrained VGG16 model from keras and modify
        the input size, number of classes, or both based on user input.
        """
        from keras.applications.vgg16 import VGG16
        from keras.layers import Input, Flatten, Dense
        from keras.models import Model
        import tensorflow as tf

        import keras
        img_dim_ordering = "channels_last"
        keras.backend.set_image_data_format(img_dim_ordering)

        inputShape = (height, width, depth)
        # Create your own input format
        input_image = Input(shape=inputShape, name='image_input')
        # Get the convolutional part of a VGG network trained on ImageNet
        model = VGG16(weights='imagenet', include_top=False, input_tensor=input_image)
        print(">ia> Original pretrained model from keras:\n")
        model.summary()

        # Use the generated model
        output_vgg16_conv = model(input_image)

        # Add the fully-connected layers
        x = Flatten(name='flatten')(output_vgg16_conv)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)

        # Create your own model
        model = tf.keras.Model(input_image, x)

        # In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
        print(">ia> Modified model using pre-trained VGG16 model from Keras:\n")
        model.summary()
        return model
