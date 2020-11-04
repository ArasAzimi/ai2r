from flask import (
    jsonify, Blueprint, request, render_template, current_app)
import os

from predict_pretrained import pretrained

bp = Blueprint('show', __name__)


@bp.route('/show', methods=['POST'])
def show():
    if request.method == 'POST':
        try:
            data = request.get_json()
            filename = data["filename"]
            image = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        except ValueError:
            return jsonify("Please enter a valid filename.")

        return render_template("show.html", in_image=image)
