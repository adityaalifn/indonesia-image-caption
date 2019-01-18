import argparse
import csv
import pickle
from math import log

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.preprocessing import sequence

from src.modeler.modeler import ImageCaptionModeler
from src.util import convert_image_to_numpy_array, get_stub_and_request, is_tensorflow_serving_running


class Predict(object):
    def __init__(self):
        pass


class InceptionV3GRUPredict(Predict):
    weight_path = 'models/InceptionV3GRU.weights.09-1.32.hdf5'

    def __init__(self, weight_path=None):
        super().__init__()
        self.tokenizer = self._load_tokenizer()
        self.weight_path = weight_path or self.weight_path
        self.load_model()

        self.start_token = self.tokenizer.word_index['<start>']
        self.end_token = self.tokenizer.word_index['<end>']

    def load_model(self):
        self.model = ImageCaptionModeler().get_model(len(self.tokenizer.word_index))
        self.model.load_weights(self.weight_path)

    def predict_batch(self, images_path, max_words=30, save_prediction_to_file=True, save_mode='caption'):
        captions, tokens = [], []
        for image_path in images_path:
            caption, token = self.predict(image_path, max_words=max_words)
            captions.append(caption)
            tokens.append(token)

        if save_prediction_to_file:
            if save_mode == 'caption':
                with open('prediction.csv', 'w') as f:
                    writer = csv.writer(f, delimiter='\t')
                    writer.writerows(zip(images_path, tokens))
            elif save_mode == 'token':
                with open('prediction.csv', 'w') as f:
                    writer = csv.writer(f, delimiter='\t')
                    writer.writerows(zip(images_path, tokens))

        return captions, tokens

    def predict_on_serving(self, image_path, max_words=30, show_image=False):
        tokenizer = InceptionV3GRUPredict().tokenizer
        try:
            image = convert_image_to_numpy_array(image_path)
        except AttributeError:
            image = image_path

        shape = (1, max_words)
        decoder_input = np.zeros(shape=shape, dtype=np.int)

        word_dict = dict(map(reversed, tokenizer.word_index.items()))
        token_int = tokenizer.word_index['start']
        predicted_caption = ""

        count_tokens = 0
        pred_word = ''
        pred_token = []

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

            pred_token.append(token_int)
            count_tokens += 1
            pred_word = word_dict[token_int]

        if show_image:
            plt.imshow(Image.open(image_path))
            plt.title(predicted_caption)
            plt.show()
        return predicted_caption

    def predict(self, image_path, max_words=30, show_image=False):
        try:
            image_arr = convert_image_to_numpy_array(image_path)
        except AttributeError:
            image_arr = image_path

        image_batch = np.expand_dims(image_arr, axis=0)
        shape = (1, max_words)
        decoder_input = np.zeros(shape=shape, dtype=np.int)

        token_int = self.start_token
        predicted_caption = ""

        word_dict = dict(map(reversed, self.tokenizer.word_index.items()))

        count_tokens = 0
        pred_word = ''
        pred_token = []
        while pred_word != '<end>' and count_tokens < max_words:
            predicted_caption += " " + pred_word

            decoder_input[0, count_tokens] = token_int
            x_data = {'input_1': image_batch,
                      'decoder_input': decoder_input}

            decoder_output = self.model.predict(x_data)

            token_one_hot = decoder_output[0, count_tokens, :]
            token_int = np.argmax(token_one_hot)

            pred_token.append(token_int)
            pred_word = word_dict[token_int]

            count_tokens += 1

        if show_image:
            plt.imshow(Image.open(image_path))
            plt.title(predicted_caption)
            plt.show()

        return predicted_caption, pred_token

    def _load_tokenizer(self):
        with open('models/tokenizer.pickle', 'rb') as handle:
            return pickle.load(handle)

    def beam_search_predictions(self, image_path, max_words=30, beam_index=3):
        image_arr = convert_image_to_numpy_array(image_path)

        start = [self.tokenizer.word_index["<start>"]]

        start_word = [[start, 0.0]]

        while len(start_word[0][0]) < max_words:
            temp = []
            for s in start_word:
                par_caps = sequence.pad_sequences([s[0]], maxlen=max_words, padding='post')
                print(par_caps)
                # e = encoding_test[image[len(images):]]
                preds = self.model.predict([np.array([image_arr]), par_caps])

                word_preds = np.argsort(preds[0])[-beam_index:]

                # Getting the top <beam_index>(n) predictions and creating a
                # new list so as to put them via the model again
                for w in word_preds:
                    next_cap, prob = s[0][:], s[1]
                    print(next_cap, prob)
                    next_cap.append(w)
                    # print(preds)
                    print(preds.shape)
                    prob += preds[0, 0, w]
                    temp.append([next_cap, prob])

            start_word = temp
            # Sorting according to the probabilities
            start_word = np.sort(start_word)
            # Getting the top words
            start_word = start_word[-beam_index:]

        start_word = start_word[-1][0]
        intermediate_caption = [self.tokenizer.word_index[i] for i in start_word]

        final_caption = []

        for i in intermediate_caption:
            if i != '<end>':
                final_caption.append(i)
            else:
                break

        final_caption = ' '.join(final_caption[1:])
        return final_caption

    def beam_search_decoder(self, image_path, k):
        image_arr = convert_image_to_numpy_array(image_path)
        tokens = [self.tokenizer.word_index['<start>']]
        sequences = [[list(), 1.0]]
        # walk over each step in sequence
        while len(tokens) < 30:
            for row in sequences:
                all_candidates = list()
                # expand each current candidate
                for i in range(len(sequences)):
                    seq, score = sequences[i]
                    for j in range(len(row)):
                        candidate = [seq + [j], score * -log(row[j])]
                        all_candidates.append(candidate)
                # order all candidates by score
                ordered = sorted(all_candidates, key=lambda tup: tup[1])
                # select k best
                sequences = ordered[:k]
                tokens.append(sequences)
        return tokens


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='image-caption Predict', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--model', default=1, type=int, help='model choice:\n'
                                                                   '1: caption an image using image caption sentence modeler\n'
                                                                   '2: caption an image using image caption single word modeler')

    parser.add_argument('-p', '--path', type=str, required=True, help='your image path')
    args = parser.parse_known_args()

    model_type = args[0].model
    image_path = args[0].path
    show_image = True if '--show-image' in args[-1] else False

    model = None
    if model_type == 1:
        model = InceptionV3GRUPredict()

    caption = model.predict_on_serving(image_path, show_image=show_image) if is_tensorflow_serving_running() \
        else model.predict(image_path, show_image=show_image)
    print(caption)
