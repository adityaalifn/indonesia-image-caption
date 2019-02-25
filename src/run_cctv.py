import logging

import cv2
from keras.applications import inception_v3
from keras.preprocessing.image import img_to_array

from src.predict import InceptionV3GRUPredict
from src.util import is_tensorflow_serving_running

logger = logging.getLogger('ImageCaption-CCTV')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

if __name__ == '__main__':
    cctv_stream = 'http://atcs-dishub.bandung.go.id/camera/PasirKalikiIP.m3u8'
    cap = cv2.VideoCapture(cctv_stream)
    frame_rate = int(cap.get(5))
    model = InceptionV3GRUPredict()
    while cap.isOpened():
        ret, frame = cap.read()
        img_array = cv2.resize(frame, (299, 299), interpolation=cv2.INTER_AREA)
        img_array = img_to_array(img_array)
        img_array = inception_v3.preprocess_input(img_array)

        if is_tensorflow_serving_running():
            caption = model.predict_on_serving(img_array)
        else:
            caption, _ = model.predict(img_array)

        cv2.putText(frame, caption, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('Pasirkaliki-Istana Plaza Street', frame)

        if cv2.waitKey(frame_rate) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
