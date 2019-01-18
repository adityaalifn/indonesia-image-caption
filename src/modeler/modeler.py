import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Embedding, GRU, RepeatVector, TimeDistributed, Bidirectional, concatenate
from keras.models import Input, Model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from src.config import CONFS


class Modeler(object):
    def __init__(self, config=CONFS, *args, **kwargs):
        self._config = config

    def get_model(self, *args, **kwargs):
        pass

    @staticmethod
    def sparse_cross_entropy(y_true, y_pred):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        loss_mean = tf.reduce_mean(loss)
        return loss_mean

    @staticmethod
    def categorical_accuracy_with_variable_timestep(y_true, y_pred):
        y_true = y_true[:, :-1, :]  # Discard the last timestep/word (dummy)
        y_pred = y_pred[:, :-1, :]  # Discard the last timestep/word (dummy)

        # Flatten the timestep dimension
        shape = tf.shape(y_true)
        y_true = tf.reshape(y_true, [-1, shape[-1]])
        y_pred = tf.reshape(y_pred, [-1, shape[-1]])

        # Discard rows that are all zeros as they represent dummy or padding words.
        is_zero_y_true = tf.equal(y_true, 0)
        is_zero_row_y_true = tf.reduce_all(is_zero_y_true, axis=-1)
        y_true = tf.boolean_mask(y_true, ~is_zero_row_y_true)
        y_pred = tf.boolean_mask(y_pred, ~is_zero_row_y_true)

        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true, axis=1),
                                                   tf.argmax(y_pred, axis=1)),
                                          dtype=tf.float32))
        return accuracy

    @staticmethod
    def bleu_score_metric(y_true, y_pred):
        # return nltk.translate.bleu_score.sentence_bleu(y_true, y_pred)
        pass


class ImageCaptionModeler(Modeler):
    def __init__(self):
        super().__init__()

    def get_model(self, vocab_size, *args, **kwargs):
        inception_v3 = InceptionV3(include_top=False, weights='imagenet', pooling='avg')
        for layer in inception_v3.layers:
            layer.trainable = False

        image_dense = Dense(512, activation='tanh', name='image_dense')(inception_v3.output)

        decoder_input = Input(shape=(None,), name='decoder_input')

        net = decoder_input
        net = Embedding(input_dim=vocab_size + 1, output_dim=128, name='decoder_embedding')(net)

        net = GRU(512, name='decoder_gru1', return_sequences=True)(net, initial_state=image_dense)
        net = GRU(512, name='decoder_gru2', return_sequences=True)(net, initial_state=image_dense)
        net = GRU(512, name='decoder_gru3', return_sequences=True)(net, initial_state=image_dense)

        decoder_output = Dense(vocab_size + 1, activation='linear', name='decoder_output')(net)

        decoder_model = Model([inception_v3.input, decoder_input],
                              outputs=[decoder_output])

        decoder_target = tf.placeholder(dtype='int32', shape=(None, None))
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        decoder_model.compile(
            optimizer=optimizer,
            loss=self.sparse_cross_entropy,
            # metrics=[self.bleu_score_metric],
            target_tensors=[decoder_target])

        return decoder_model


class ImageCaptionGruModeler(Modeler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_model(self, vocab_size=8495, *args, **kwargs):
        inception_v3 = InceptionV3(include_top=False, weights='imagenet', pooling='avg')
        for layer in inception_v3.layers:
            layer.trainable = False

        image_embedding = Dense(300, activation='relu', name='image_dense')(inception_v3.output)
        image_embedding = RepeatVector(40)(image_embedding)

        input_caption = Input(shape=(None,), name='input_caption')
        caption_encoder = Embedding(vocab_size, output_dim=300, input_length=40)(input_caption)
        caption_encoder = GRU(256, return_sequences=True)(caption_encoder)
        caption_encoder = GRU(256, return_sequences=True)(caption_encoder)
        caption_encoder = TimeDistributed(Dense(300))(caption_encoder)

        net = concatenate([image_embedding, caption_encoder], axis=1)
        net = Bidirectional(GRU(256, return_sequences=False))(net)
        caption_output = Dense(vocab_size, activation='softmax', name='caption_output')(net)

        image_caption_model = Model([inception_v3.input, input_caption], outputs=caption_output)

        try:
            image_caption_model = multi_gpu_model(image_caption_model)
        except:
            pass

        image_caption_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        return image_caption_model


if __name__ == '__main__':
    print(ImageCaptionModeler().get_model(10000).summary())
