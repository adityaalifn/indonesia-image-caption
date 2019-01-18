from keras.callbacks import ModelCheckpoint, TensorBoard

from src.config import CONFS
from src.constant import DatasetKeys


class Callback():
    def __init__(self, model_name, config=CONFS):
        self._config = config
        self._model_name = model_name
        self.callbacks = []
        self._batch_size = self._config[DatasetKeys.DATASET][DatasetKeys.FLICKR8K][DatasetKeys.BATCH_SIZE]
        self.build()

    def build(self):
        self.callbacks.append(
            ModelCheckpoint('models/' + self._model_name + '.weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                            save_best_only=True,
                            verbose=1))
        self.callbacks.append(TensorBoard(batch_size=self._batch_size))
