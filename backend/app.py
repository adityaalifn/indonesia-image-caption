import base64
import io
import json
import os

from PIL import Image
from flask import Flask, request

from src.predict import predict

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route('/')
def index():
    return 'path not available'


@app.route('/api/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        image_byte = request.data
        if image_byte == '':
            return 'Format not supported'
        image = base64.b64decode(image_byte)
        image = io.BytesIO(image)
        image = Image.open(image)
        image.thumbnail((299, 299), Image.ANTIALIAS)

        caption = predict(image)
        response = app.response_class(
            response=json.dumps({'caption': caption}),
            status=200,
            mimetype='application/json'
        )
        return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)
