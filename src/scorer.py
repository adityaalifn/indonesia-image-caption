import sys

from nltk.translate.bleu_score import sentence_bleu

from src.config import CONFS
from src.constant import DatasetKeys
from src.dataset import Flickr8kDataset
from src.predict import InceptionV3GRUPredict


class Score(object):
    def __init__(self):
        pass

    def get_score(self, target_caption, predicted_caption):
        pass

    def print_progress_bar(self, count, total, status=''):
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
        sys.stdout.flush()


class BleuScore(Score):
    def __init__(self):
        super().__init__()

    def get_score(self, target_caption, predicted_caption):
        bleu1, bleu2, bleu3, bleu4 = 0, 0, 0, 0
        if len(predicted_caption) >= 1:
            bleu1 = sentence_bleu(references=target_caption, hypothesis=predicted_caption, weights=(1, 0, 0, 0))

        if len(predicted_caption) >= 2:
            bleu2 = sentence_bleu(references=target_caption, hypothesis=predicted_caption, weights=(0.5, 0.5, 0, 0))

        if len(predicted_caption) >= 3:
            bleu3 = sentence_bleu(references=target_caption, hypothesis=predicted_caption,
                                  weights=(0.33, 0.33, 0.33, 0))

        if len(predicted_caption) >= 4:
            bleu4 = sentence_bleu(references=target_caption, hypothesis=predicted_caption,
                                  weights=(0.25, 0.25, 0.25, 0.25))

        return bleu1, bleu2, bleu3, bleu4

    def get_model_score(self, save_score_to_file=True, weight_path=None, tokenizer_path=None, language="english", save_result=False):
        dataset = Flickr8kDataset(language=language)
        filenames, captions = dataset.get_test_dataset()
        img_folder_path = CONFS[DatasetKeys.DATASET][DatasetKeys.FLICKR30K][DatasetKeys.IMAGE_FOLDER]
        model = InceptionV3GRUPredict(weight_path, tokenizer_path=tokenizer_path, language=language)

        # count = 0
        target_captions = []
        bleu1, bleu2, bleu3, bleu4 = 0, 0, 0, 0
        print("[INFO] SCORING IN " + str(len(filenames)) + " IMAGES")
        for filename, caption in zip(filenames, captions):
            target_captions.append(caption.translate(str.maketrans('', '', '!"#$%&()*+,./:;=?@[\]^_`{|}~')).split())
            if len(target_captions) >= 5:
                pred_caption = model.predict(img_folder_path + '/' + filename, save_result=True)[0].split()
                print("------------------------------")
                print(" ".join(pred_caption))
                print(target_captions)
                temp_bleu1, temp_bleu2, temp_bleu3, temp_bleu4 = self.get_score(target_captions, pred_caption)
                bleu1 += temp_bleu1
                bleu2 += temp_bleu2
                bleu3 += temp_bleu3
                bleu4 += temp_bleu4

                target_captions = []
                # count += 1
                # self.print_progress_bar(count, 1000, 'BLEU scoring is in progress...')

        if save_score_to_file:
            weight_path = model.weight_path
            with open('score.txt', 'a') as text_file:
                print(f'({weight_path}) Bleu Score: {bleu1 / 1000} {bleu2 / 1000} {bleu3 / 1000} {bleu4 / 1000}',
                      file=text_file)

        return bleu1 / 1000, bleu2 / 1000, bleu3 / 1000, bleu4 / 1000


class CiderScore(Score):
    def __init__(self):
        super().__init__()


if __name__ == '__main__':
    bs = BleuScore()
    print(
        bs.get_model_score(save_score_to_file=True,
                           weight_path='models/InceptionV3GRU_decoder_gru2.weights.01-0.91_indonesia.hdf5',
                           tokenizer_path='models/tokenizer_indonesia_2019-03-10_19h19m14s.pickle',
                           language="indonesia", save_result=True))
