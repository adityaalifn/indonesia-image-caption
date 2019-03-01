import pickle

import numpy as np
from keras.preprocessing import sequence

from src.config import CONFS
from src.constant import DatasetKeys
from src.dataset import Flickr8kDataset, Flickr8kSingleWordDataset
from src.preprocessor import CaptionsPreprocessor, ImagePreprocessor
from src.util import shuffle_dict, shuffle_array, convert_image_to_numpy_array


class Generator(object):
    def __init__(self, config=CONFS, batch_size=None):
        self._config = config
        self._batch_size = batch_size or self._config[DatasetKeys.DATASET][DatasetKeys.FLICKR8K][DatasetKeys.BATCH_SIZE]

    def train_generator(self):
        pass

    def validation_generator(self):
        pass

    def test_generator(self):
        pass


class Flickr8kGenerator(Generator):
    def __init__(self, config=CONFS, batch_size=None, language="english"):
        super().__init__(config, batch_size)
        self.dataset = Flickr8kDataset(language=language)
        self.train_filenames, self.train_captions = self.dataset.get_train_dataset()
        self.validation_filenames, self.validation_captions = self.dataset.get_validation_dataset()
        self.test_filenames, self.test_captions = self.dataset.get_test_dataset()
        self.ip = ImagePreprocessor(image_augmentation=False)
        self.cp = CaptionsPreprocessor(captions=self.train_captions + self.validation_captions)
        self._reset()

        with open('models/tokenizer_' + language + '.pickle', 'wb') as handle:
            pickle.dump(self.cp.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def train_generator(self):
        yield from self._batch_generator(self.train_filenames, self.train_captions)

    def validation_generator(self):
        yield from self._batch_generator(self.validation_filenames, self.validation_captions)

    def test_generator(self):
        yield from self._batch_generator(self.test_filenames, self.test_captions)

    def _batch_generator(self, filenames, captions):
        self._reset()
        while True:
            for filename, caption in zip(filenames, captions):
                self.count += 1
                self.batch_captions.append(caption)
                self.batch_filenames.append(filename)
                if self.count >= self._batch_size:
                    yield from self._preprocess_batch(self.batch_filenames, self.batch_captions)
                    self._reset()

            if self.batch_captions:
                yield from self._preprocess_batch(self.batch_filenames, self.batch_captions)
                self._reset()

    def _preprocess_batch(self, batch_filenames, batch_captions):
        img_folder_path = self._config[DatasetKeys.DATASET][DatasetKeys.FLICKR8K][DatasetKeys.IMAGE_FOLDER]
        batch_img_path = [img_folder_path + filename for filename in batch_filenames]
        batch_img_arr = [self.ip.preprocess_image(img_path) for img_path in batch_img_path]

        input_sequence, output_sequence = self.cp.preprocess_caption(batch_captions)
        X, y = [np.array(batch_img_arr), input_sequence], output_sequence
        yield X, y

    def _reset(self):
        self.batch_captions = []
        self.batch_filenames = []
        self.count = 0


class Flickr8kSingleWordGenerator(Generator):
    def __init__(self,
                 config=CONFS, shuffle=True, language="english"):
        self._config = config
        self._batch_size = self._config[DatasetKeys.DATASET][DatasetKeys.FLICKR8K][DatasetKeys.BATCH_SIZE]
        self.dataset = Flickr8kSingleWordDataset(language=language)
        self.shuffle = shuffle
        self.dataset.build()
        self.tokenizer = self.dataset.get_tokenizer()
        self.vocab_size = len(self.tokenizer.word_counts)
        self._reset()
        super().__init__(config)

        with open('models/tokenizer_sw_' + language + '.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def train_generator(self):
        self._reset()
        _, train_dataset_dictionaries_mask = self.dataset.get_train_dataset()
        while True:
            train_dataset_dictionaries_mask = shuffle_dict(train_dataset_dictionaries_mask)
            yield from self.preprocess_batch(train_dataset_dictionaries_mask)
            self._reset()

    def validation_generator(self):
        self._reset()
        _, validation_dataset_dictionaries_mask = self.dataset.get_validation_dataset()
        while True:
            validation_dataset_dictionaries_mask = shuffle_dict(validation_dataset_dictionaries_mask)
            yield from self.preprocess_batch(validation_dataset_dictionaries_mask)
            self._reset()

    def test_generator(self):
        self._reset()
        _, test_dataset_dictionaries_mask = self.dataset.get_test_dataset()
        while True:
            test_dataset_dictionaries_mask = shuffle_dict(test_dataset_dictionaries_mask)
            yield from self.preprocess_batch(test_dataset_dictionaries_mask)
            self._reset()

    def preprocess_batch(self, dictionary):
        for image_name, sentences in dictionary.items():
            current_image = convert_image_to_numpy_array(
                self._config[DatasetKeys.DATASET][DatasetKeys.FLICKR8K][DatasetKeys.IMAGE_FOLDER] +
                image_name.split('#')[0],
                target_size=(299, 299))
            yield from self._preprocess_sentences(sentences, current_image)

    def _preprocess_sentences(self, sentences, current_image):
        for sentence in sentences:
            sentence = [sentence]
            sentence_length = len(self.tokenizer.texts_to_sequences(sentence)[0]) - 1
            sentence_sequence = self.tokenizer.texts_to_sequences(sentence)[0]
            for index in range(sentence_length):
                try:
                    self.partial_captions.append(sentence_sequence[:index + 1])
                except AttributeError:
                    self.partial_captions = self.partial_captions.tolist()
                    self.partial_captions.append(sentence_sequence[:index + 1])
                self.count += 1
                one_hot_numpy_array = np.zeros(self.vocab_size)
                one_hot_numpy_array[sentence_sequence[index + 1] - 1] = 1
                self.next_word.append(one_hot_numpy_array)
                self.images.append(current_image)
                if self.count >= self._batch_size:
                    self.partial_captions_sequence = sequence.pad_sequences(self.partial_captions, maxlen=40, padding='post')
                    indices = list(range(len(self.next_word)))
                    indices = shuffle_array(indices) if self.shuffle else indices
                    yield [[np.array(self.images)[indices], self.partial_captions_sequence[indices]],
                           np.array(self.next_word)[indices]]
                    self._reset()

    def _reset(self):
        self.images = []
        self.partial_captions = []
        self.next_word = []
        self.count = 0
