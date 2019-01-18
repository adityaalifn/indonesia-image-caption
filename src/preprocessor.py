from keras.applications import inception_v3
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


class Preprocessor(object):
    def __init__(self):
        pass


class CaptionsPreprocessor(Preprocessor):
    input_sequences = []
    output_sequences = []

    def __init__(self,
                 captions,
                 num_words=10000,
                 filters=Tokenizer().filters.replace("<", "").replace(">", "")):
        super().__init__()
        self.tokenizer = Tokenizer(num_words=num_words,
                                   filters=filters)
        self.captions = self.mark_captions(captions)
        self.tokenizer.fit_on_texts(texts=self.captions)
        self.vocab_size = len(self.tokenizer.word_index)

    def mark_captions(self, captions):
        start_mark = '<start> '
        end_mark = ' <end>'

        marked_captions = [start_mark + caption + end_mark for caption in captions]
        return marked_captions

    def captions_to_sequences(self, captions):
        return self.tokenizer.texts_to_sequences(captions)

    @staticmethod
    def pad_sequences(sequences, maxlen=30):
        return pad_sequences(sequences, maxlen=maxlen, padding="post", truncating="post")

    def shift_sequences(self, sequences):
        input_sequences = sequences[:, 0:-1]
        output_sequences = sequences[:, 1:]

        return input_sequences, output_sequences

    def build_sequences(self):
        sequences = self.captions_to_sequences(self.captions)
        padded_sequences = self.pad_sequences(sequences)
        self.shift_sequences(padded_sequences)
        return self.input_sequences, self.output_sequences

    def preprocess_caption(self, captions):
        marked_captions = self.mark_captions(captions)
        sequences = self.captions_to_sequences(marked_captions)
        padded_sequences = self.pad_sequences(sequences)
        input_sequences, output_sequences = self.shift_sequences(padded_sequences)
        return input_sequences, output_sequences


class ImagePreprocessor(Preprocessor):
    _IMAGE_SIZE = (299, 299)

    def __init__(self, image_augmentation=True):
        super().__init__()
        self._image_augmentation = image_augmentation
        self._image_data_generator = ImageDataGenerator(rotation_range=40,
                                                        width_shift_range=0.2,
                                                        height_shift_range=0.2,
                                                        shear_range=0.2,
                                                        zoom_range=0.2,
                                                        horizontal_flip=True,
                                                        fill_mode='nearest')

    def preprocess_images(self, img_paths):
        return map(self.preprocess_image, img_paths)

    def preprocess_image(self, img_path):
        img = load_img(img_path, target_size=self._IMAGE_SIZE)
        img_array = img_to_array(img)
        if self._image_augmentation:
            img_array = self._image_data_generator.random_transform(img_array)
        img_array = inception_v3.preprocess_input(img_array)

        return img_array
