import os
from flask import Flask

def create_app(test_config=None):
    # Create and configure the app
    app = Flask(__name__, instance_relative_config=True)


    AIRCRAFTS_FOLDER = os.path.join('static', 'aircrafts')
    app.config['UPLOAD_FOLDER'] = AIRCRAFTS_FOLDER

    # Ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # A simple page that says hello to a user!
    @app.route('/hello/<string:user>')
    def hello(user=None):
        return 'Hello {}!'.format(user)


    from api import (
         show)

    # Register lueprints to the app
    app.register_blueprint(show.bp)

    return app
