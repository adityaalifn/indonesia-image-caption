import numpy as np
import tensorflow as tf
from keras.preprocessing.image import img_to_array

from src.util import get_tokenizer, get_stub_and_request


def predict(image, max_words=30):
    tokenizer = get_tokenizer()
    # image = convert_image_byte_to_numpy_array(image_path, target_size=(299, 299))
    image = img_to_array(image)
    shape = (1, max_words)
    decoder_input = np.zeros(shape=shape, dtype=np.int)

    word_dict = dict(map(reversed, tokenizer.word_index.items()))
    token_int = tokenizer.word_index['start']
    predicted_caption = ""

    count_tokens = 0
    pred_word = ''

    stub, request = get_stub_and_request()
    while pred_word != '<end>' and count_tokens < max_words:
        predicted_caption += " " + pred_word

        decoder_input[0, count_tokens] = token_int

        x_data = {'image': np.expand_dims(image, axis=0),
                  'sequence_input': decoder_input}

        for k, v in x_data.items():
            request.inputs[k].CopyFrom(
                tf.contrib.util.make_tensor_proto(
                    v,
                    shape=v.shape,
                    dtype=tf.float32
                ))

        result = stub.Predict(request, 5)
        array_result = np.array(result.outputs['sequence_output'].float_val)
        token_one_hot = np.reshape(array_result, (1, max_words, len(tokenizer.word_index) + 1))
        token_int = np.argmax(token_one_hot[0, count_tokens, :])

        count_tokens += 1
        pred_word = word_dict[token_int]

    return predicted_caption
