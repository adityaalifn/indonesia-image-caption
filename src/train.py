import argparse
import datetime
import os
import pickle

from src.config import CONFS
from src.constant import DatasetKeys
from src.generator import Flickr8kGenerator, Flickr8kSingleWordGenerator
from src.modeler.callback import Callback
from src.modeler.modeler import ImageCaptionModeler, ImageCaptionGruModeler
from src.serving import KerasServing


class Train(object):
    def __init__(self, config):
        self._tokenizer = None
        self._config = config
        self._batch_size = self._config[DatasetKeys.DATASET][DatasetKeys.FLICKR8K][DatasetKeys.BATCH_SIZE]
        self._serving = KerasServing()

    def save_tokenizer(self, filename):
        with open(filename, 'wb') as handle:
            pickle.dump(self._tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


class TrainInceptionV3GRU(Train):
    def __init__(self, config=CONFS, existing_model_path='', train_generator=None, validation_generator=None,
                 language="english"):
        super().__init__(config=config)
        self.language = language
        self._generator = Flickr8kGenerator(language=language)
        self._train_generator = train_generator or self._generator.train_generator()
        self._validation_generator = validation_generator or self._generator.validation_generator()
        self._train_data_amount = len(self._generator.train_captions)
        self._validation_data_amount = len(self._generator.validation_captions)
        self._tokenizer = self._generator.cp.tokenizer
        self.model = ImageCaptionModeler().get_model(self._generator.cp.vocab_size)

        # load previous model if exist
        if os.path.exists(existing_model_path):
            self.model.load_weights(existing_model_path)

    def run(self):
        callback = Callback('InceptionV3GRU')

        self.model.fit_generator(
            generator=self._train_generator,
            steps_per_epoch=self._train_data_amount // self._batch_size,
            epochs=10,
            validation_data=self._validation_generator,
            validation_steps=16,
            callbacks=callback.callbacks
        )

        cur_time = datetime.datetime.now()
        self.model.save('models/InceptionV3GRU_model_' + str(cur_time) + '_' + self.language + "_" + self.model.layers[
            -2].name + '.h5')
        self.model.save_weights(
            'models/InceptionV3GRU_weight_' + str(cur_time) + '_' + self.language + "_" + self.model.layers[
                -2].name + '.h5')
        self.save_tokenizer('models/tokenizer_' + self.language + '_' + str(cur_time) + '.pickle')

        # Save model for serving
        self._serving.save_model(self.model)


class ImageCaptionSingleWordTrain(Train):
    def __init__(self, existing_model_path=''):
        self._generator = Flickr8kSingleWordGenerator()
        self._tokenizer = self._generator.tokenizer
        self._train_generator = self._generator.train_generator()
        self._validation_generator = self._generator.validation_generator()
        self.model = ImageCaptionGruModeler().get_model(vocab_size=len(self._tokenizer.word_index))
        super().__init__(CONFS)

        # load previous model if exist
        if os.path.exists(existing_model_path):
            self.model.load_weights(existing_model_path)

    def run(self):
        callback = Callback('ImageCaptionSingleWord')
        self.model.fit_generator(generator=self._train_generator,
                                 steps_per_epoch=6000 * 5 * 13 // self._batch_size,
                                 epochs=20,
                                 validation_data=self._validation_generator,
                                 validation_steps=16,
                                 callbacks=callback.callbacks)

        self.model.save('models/ImageCaptionSingleWord_model.h5')
        self.model.save_weights('models/ImageCaptionSingleWord_weights.h5')
        self.save_tokenizer('models/tokenizerSingleWord.pickle')
        self._serving.save_model(self.model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='image-caption Train', formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-m', '--model', type=int, default=1, help='model choice:\n'
                                                                   '1: train image caption sentence modeler\n'
                                                                   '2: train image caption single word modeler')
    args = parser.parse_args()

    if args.model == 1:
        train = TrainInceptionV3GRU(language="indonesia")
        train.run()
    elif args.model == 2:
        train = ImageCaptionSingleWordTrain()
        train.run()
