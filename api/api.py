import os
from flask import Flask
from flask_restful import Resource, Api

def create_app(test_config=None):
    """
    Create the flask app and initialize some required configurations
    and relationships
    """
    # Create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    api = Api(app)

    AIRCRAFTS_FOLDER = os.path.join('static', 'aircrafts')
    TRAINED_MODEL_FOLDER = os.path.join(os.path.split(os.getcwd())[0],
                                        'deployment', 'models')
    CONFIG_FILE = os.path.join(os.path.split(os.getcwd())[0], 'config.json')
    DATABASE=os.path.join(app.instance_path, 'api.sqlite')

    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=DATABASE,
        UPLOAD_FOLDER = AIRCRAFTS_FOLDER,
        MODEL_FOLDER = TRAINED_MODEL_FOLDER,
        MODEL_CONFIG_FILE = CONFIG_FILE
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # A simple page that says hello
    @app.route('/')
    def hello(user=None):
        return 'Hello. Welcome to ai2r homepage!'.format(user)

    # Apply the blueprints to the app
    import show

    from resources.train import TrainResource
    from resources.predict import PredictResource

    api.add_resource(TrainResource, '/train')
    api.add_resource(PredictResource, '/predict')

    app.register_blueprint(show.bp)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
