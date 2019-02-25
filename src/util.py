import base64
import pickle

import docker
import numpy as np
from grpc.beta import implementations
from keras.applications import inception_v3
from keras.preprocessing.image import img_to_array, load_img
from tensorflow_serving.apis import prediction_service_pb2, predict_pb2


def shuffle_dict(dictionaries):
    keys = list(dictionaries.keys())
    np.random.shuffle(keys)
    return {key: dictionaries[key] for key in keys}


def shuffle_array(array):
    np.random.shuffle(array)
    return array


def convert_image_to_numpy_array(path, target_size=(299, 299)):
    img = load_img(path, target_size=target_size)
    img_array = img_to_array(img)
    return inception_v3.preprocess_input(img_array)


def get_stub_and_request():
    stub = prediction_service_pb2.beta_create_PredictionService_stub(implementations.insecure_channel(
        'localhost',
        int(8500)
    ))

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'inception'
    request.model_spec.signature_name = 'predict'

    return stub, request


def get_tokenizer():
    with open('models/tokenizer.pickle', 'rb') as handle:
        return pickle.load(handle)


def decode_image(image_byte):
    return base64.b64encode(image_byte)


def is_tensorflow_serving_running():
    client = docker.from_env()
    container = client.containers.get('model-image_caption')
    if container in client.containers.list():
        return True
    else:
        return False
