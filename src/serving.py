import argparse
import time

import tensorflow as tf
from keras import backend as K
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

from src.predict import InceptionV3GRUPredict


class Serving(object):
    def __init__(self):
        pass


class KerasServing(Serving):
    def __init__(self):
        super().__init__()

    def save_model(self, model, input='array'):
        K.set_learning_phase(0)

        input_caption = model.input[1]
        input_image = model.input[0]
        output_prediction = model.output

        if input == 'bytes':
            input_image = tf.placeholder(tf.string, shape=(None,), name='input_string')
            input_bytes_map = tf.map_fn(tf.decode_base64, input_image)
            input_bytes_map.set_shape((None,))
            input_tensor_map = tf.map_fn(tf.image.decode_image, input_bytes_map, dtype=tf.float32)
            input_tensor_map.set_shape((None, None, None, 3))
            input_tensor = tf.image.convert_image_dtype(input_tensor_map, dtype=tf.float32)
            output_prediction = model([input_tensor, input_caption])

        export_path = 'models/serving_model/model-data/{timestamp}'.format(timestamp=int(time.time()))
        builder = saved_model_builder.SavedModelBuilder(export_path)

        signature = predict_signature_def(inputs={'image': input_image,
                                                  'sequence_input': input_caption},
                                          outputs={'sequence_output': output_prediction})

        with K.get_session() as sess:
            builder.add_meta_graph_and_variables(sess=sess,
                                                 tags=[tag_constants.SERVING],
                                                 signature_def_map={'predict': signature})
            builder.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='image-caption Serving', formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-m', '--model', type=int, default=1, help='model choice:\n'
                                                                   '1: serve image caption sentence modeler\n'
                                                                   '2: serve image caption single word modeler')
    parser.add_argument('-p', '--path', type=str, help='input your keras model path', required=True)
    args = parser.parse_args()

    model_path = args.path
    model_type = args.model

    ks = KerasServing()
    if model_type == 1:
        model = InceptionV3GRUPredict(weight_path=model_path).model
        ks.save_model(model)
    elif model_type == 2:
        from src.train import ImageCaptionSingleWordTrain

        model = ImageCaptionSingleWordTrain(existing_model_path=model_path).model
        ks.save_model(model)
