import pickle

import numpy as np
from grpc.beta import implementations
from keras.applications import inception_v3
from keras.preprocessing.image import img_to_array
from tensorflow_serving.apis import prediction_service_pb2, predict_pb2


def shuffle_array(array):
    np.random.shuffle(array)
    return array


def convert_pil_image_to_numpy_array(pil_image, target_size=(299, 299)):
    img_array = img_to_array(pil_image)
    return inception_v3.preprocess_input(img_array)


def get_stub_and_request():
    stub = prediction_service_pb2.beta_create_PredictionService_stub(implementations.insecure_channel(
        '10.5.0.2',
        int(8500)
    ))

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'inception'
    request.model_spec.signature_name = 'predict'

    return stub, request


def get_tokenizer():
    with open('static/tokenizer.pickle', 'rb') as handle:
        return pickle.load(handle)
