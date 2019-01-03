from flask import (
    jsonify, Blueprint, request, render_template)
import os


bp = Blueprint('show', __name__)

@bp.route('/show', methods=['GET', 'POST'])
def show():
    if request.method == 'POST':
        try:
            data = request.get_json()
            filename = data["filename"]
            full_filename = os.path.join('static', 'aircrafts', filename)

        except ValueError:
            return jsonify("Please enter a valid filename.")

        return render_template("show.html", in_image = full_filename)
