from flask_restful import Resource, reqparse

class TrainResource(Resource):
    """
    Flask resource for training
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
        return{self.model_name:self.data_name}

    def get(self):
        return{self.model_name:self.data_name}

    def put(self):
        pass

    def delete(self):
        pass
