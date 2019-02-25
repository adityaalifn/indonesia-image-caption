import json
import os
from datetime import datetime
from functools import wraps, update_wrapper

import requests
from flask import Flask, render_template, request, make_response
from werkzeug.utils import secure_filename

from src.util import convert_image_to_base64

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response

    return update_wrapper(no_cache, view)


@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route("/index")
@app.route("/")
@nocache
def index():
    return render_template("home.html", file_path="img/your_image_here.jpg")


@app.route("/about")
@nocache
def about():
    pass


@app.route("/upload", methods=["POST"])
@nocache
def upload():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("home.html", file_path="img/no_image_selected.gif")

        file = request.files["file"]
        if file.filename == "":
            return render_template("home.html", file_path="img/no_image_selected.gif")

        if file:
            filename = secure_filename(file.filename)
            path = os.path.join(APP_ROOT, "static/img/" + filename)
            file.save(path)
            image_byte = convert_image_to_base64(path)
            response = requests.post('http://10.5.0.4:5050/api/predict', data=image_byte)
            response_content = json.loads(response.text)
            return render_template("uploaded.html", file_path="img/" + filename,
                                   caption=str(response_content['caption']))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
