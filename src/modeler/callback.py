import math

from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler

from src.config import CONFS
from src.constant import DatasetKeys


class Callback:
    def __init__(self, model_name, language="english", config=CONFS):
        self._config = config
        self._model_name = model_name
        self.callbacks = []
        self.language = language
        self._batch_size = self._config[DatasetKeys.DATASET][DatasetKeys.FLICKR8K][DatasetKeys.BATCH_SIZE]
        self.build()

    def build(self):
        self.callbacks.append(
            ModelCheckpoint(
                'models/' + self._model_name + '.weights.{epoch:02d}-{val_loss:.2f}_' + self.language + '.hdf5',
                save_best_only=False,
                verbose=1))
        self.callbacks.append(
            TensorBoard(batch_size=self._batch_size, log_dir='logs/tensorboard', write_graph=True, write_images=True))
        self.callbacks.append(LearningRateScheduler(self.step_decay))

    def step_decay(self, epoch):
        initial_lrate = 0.001
        drop = 0.5
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lrate
