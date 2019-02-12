import itertools

from keras.preprocessing.text import Tokenizer
from tensorflow.python.lib.io import file_io

from src.config import CONFS
from src.constant import DatasetKeys


class Dataset(object):
    def __init__(self, config=CONFS):
        self._config = config
        self.tokenizer = Tokenizer(filters='!"#$%&()*+,./:;=?@[\]^_`{|}~ ')

    def build(self):
        pass

    def get_train_dataset(self):
        pass

    def get_validation_dataset(self):
        pass

    def get_test_dataset(self):
        pass


class Flickr8kDataset(Dataset):
    def __init__(self, config=CONFS, language='english'):
        self.train_filenames = []
        self.validation_filenames = []
        self.test_filenames = []
        self.captions = {}
        self.language = language
        super().__init__(config)
        self.build()

    def build(self):
        self._load_filenames()
        self._load_captions()

    def _load_filenames(self):
        with open(self._config[DatasetKeys.DATASET][DatasetKeys.FLICKR8K][DatasetKeys.TRAIN_FILE],
                  'r') as file:
            self.train_filenames = file.read().split('\n')[:-1]

        with open(self._config[DatasetKeys.DATASET][DatasetKeys.FLICKR8K][DatasetKeys.VALIDATION_FILE],
                  'r') as file:
            self.validation_filenames = file.read().split('\n')[:-1]

        with open(self._config[DatasetKeys.DATASET][DatasetKeys.FLICKR8K][DatasetKeys.TEST_FILE], 'r') as file:
            self.test_filenames = file.read().split('\n')[:-1]

    def _load_captions(self):
        caption_filename = self._config[DatasetKeys.DATASET][DatasetKeys.FLICKR8K][
            DatasetKeys.CAPTION_TOKEN_FILE_ENGLISH]
        if self.language == "indonesia":
            caption_filename = self._config[DatasetKeys.DATASET][DatasetKeys.FLICKR8K][
                DatasetKeys.CAPTION_TOKEN_FILE_INDONESIA]
        with open(caption_filename, 'r') as file:
            for row in file.read().split('\n')[:-1]:
                row = row.split('\t')
                file_name = row[0].split('#')[0]
                if file_name in self.captions:
                    self.captions[file_name].append(row[1])
                else:
                    self.captions[file_name] = [row[1]]

    def get_train_dataset(self):
        train_filename_dataset = []
        train_caption_dataset = []
        for filename in self.train_filenames:
            for caption in self.captions[filename]:
                train_caption_dataset.append(caption)
                train_filename_dataset.append(filename)
        return train_filename_dataset, train_caption_dataset

    def get_validation_dataset(self):
        validation_filename_dataset = []
        validation_caption_dataset = []
        for filename in self.validation_filenames:
            for caption in self.captions[filename]:
                validation_caption_dataset.append(caption)
                validation_filename_dataset.append(filename)
        return validation_filename_dataset, validation_caption_dataset

    def get_test_dataset(self):
        test_filename_dataset = []
        test_caption_dataset = []
        for filename in self.test_filenames:
            for caption in self.captions[filename]:
                test_caption_dataset.append(caption)
                test_filename_dataset.append(filename)

        return test_filename_dataset, test_caption_dataset


class Flickr8kSingleWordDataset(Dataset):
    def __init__(self, language="english"):
        self._dictionary = {}
        self._dictionary_marks = {}
        super().__init__()
        self.language = language

    def build(self):
        self._create_dataset()
        self._build_tokenizer()

    def _create_dataset(self):
        caption_filename = self._config[DatasetKeys.DATASET][DatasetKeys.FLICKR8K][
            DatasetKeys.CAPTION_TOKEN_FILE_ENGLISH]
        if self.language == "indonesia":
            caption_filename = self._config[DatasetKeys.DATASET][DatasetKeys.FLICKR8K][
                DatasetKeys.CAPTION_TOKEN_FILE_INDONESIA]

        with file_io.FileIO(caption_filename, 'r') as file:
            captions = file.read().strip().split('\n')

        _start_mark = self._config[DatasetKeys.DATASET][DatasetKeys.FLICKR8K][DatasetKeys.START_MARK]
        _end_mark = self._config[DatasetKeys.DATASET][DatasetKeys.FLICKR8K][DatasetKeys.END_MARK]

        for index, caption in enumerate(captions):
            row = caption.split('\t')
            image_name = row[0]
            if image_name in self._dictionary:
                self._dictionary[image_name].append(row[1])
                self._dictionary_marks[image_name].append(_start_mark + row[1] + _end_mark)
            else:
                self._dictionary[image_name] = [row[1]]
                self._dictionary_marks[image_name] = [_start_mark + row[1] + _end_mark]

    def _build_tokenizer(self):
        sentences = list(itertools.chain.from_iterable(self._dictionary_marks.values()))
        self.tokenizer.fit_on_texts(sentences, )

    def get_train_dataset(self):
        with file_io.FileIO(self._config[DatasetKeys.DATASET][DatasetKeys.FLICKR8K][DatasetKeys.TRAIN_FILE],
                            'r') as file:
            train_images = file.read().strip().split('\n')[:-1]

            train_images_extended = []
            for train_image in train_images:
                for index in range(5):
                    train_images_extended.append(train_image + '#{}'.format(index))

        return ({key: self._dictionary[key] for key in train_images_extended},
                {key: self._dictionary_marks[key] for key in train_images_extended})

    def get_validation_dataset(self):
        with file_io.FileIO(self._config[DatasetKeys.DATASET][DatasetKeys.FLICKR8K][DatasetKeys.VALIDATION_FILE],
                            'r') as file:
            validation_images = file.read().strip().split('\n')[:-1]

            validation_images_extended = []
            for validation_image in validation_images:
                for index in range(5):
                    validation_images_extended.append(validation_image + '#{}'.format(index))

        return ({key: self._dictionary[key] for key in validation_images_extended},
                {key: self._dictionary_marks[key] for key in validation_images_extended})

    def get_test_dataset(self):
        with file_io.FileIO(self._config[DatasetKeys.DATASET][DatasetKeys.FLICKR8K][DatasetKeys.TEST_FILE],
                            'r') as file:
            test_images = file.read().strip().split('\n')[:-1]

            test_images_extended = []
            for test_image in test_images:
                for index in range(5):
                    test_images_extended.append(test_image + '#{}'.format(index))

        return ({key: self._dictionary[key] for key in test_images_extended},
                {key: self._dictionary_marks[key] for key in test_images_extended})

    def get_tokenizer(self):
        return self.tokenizer


class Flickr8kIndonesiaDataset(Dataset):
    def __init__(self):
        super().__init__()
