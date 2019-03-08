

import sys
import cv2
import dlib
import numpy as np
import face_recognition
from statistics import mode
from keras.models import model_from_json
from statistics import StatisticsError
from EmotionAnalys.utils.datasets import get_labels
from EmotionAnalys.utils.preprocessor import preprocess_input
from PIL import Image, ImageEnhance

# ----- Facial Emotion Recognition ----------------------------------------------------------------------------- START -

# parameters for loading data and images
emotion_model_path = 'EmotionAnalys/models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')

# loading models
# emotion_classifier = load_model(emotion_model_path)

with open("EmotionAnalys/models/emotion_json_model.json", 'r') as json_file:
    loaded_json = json_file.read()
emotion_classifier = model_from_json(loaded_json)
emotion_classifier.load_weights("EmotionAnalys/models/weight_model.h5")

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# ----- Facial Emotion Recognition ------------------------------------------------------------------------------- END -

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('EmotionAnalys/models/shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)

def reduce_opacity(im, opacity):
    """Returns an image with reduced opacity."""
    assert opacity >= 0 and opacity <= 1
    if im.mode != 'RGBA':
        im = im.convert('RGBA')
    else:
        im = im.copy()
    alpha = im.split()[3]
    alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
    im.putalpha(alpha)
    return im

def watermark(im, mark, position, opacity=1):
    """Adds a watermark to an image."""
    if opacity < 1:
        mark = reduce_opacity(mark, opacity)
    if im.mode != 'RGBA':
        im = im.convert('RGBA')
    # create a transparent layer the size of the image and draw the
    # watermark in that layer.
    layer = Image.new('RGBA', im.size, (0, 0, 0, 0))
    if position == 'tile':
        for y in range(0, im.size[1], mark.size[1]):
            for x in range(0, im.size[0], mark.size[0]):
                layer.paste(mark, (x, y))

    elif position == 'scale':
        # scale, but preserve the aspect ratio
        ratio = min(
            float(im.size[0]) / mark.size[0], float(im.size[1]) / mark.size[1])
        w = int(mark.size[0] * ratio)
        h = int(mark.size[1] * ratio)
        mark = mark.resize((w, h))
        layer.paste(mark, (int((im.size[0] - w) / 2), int((im.size[1] - h) / 2)))
    else:
        layer.paste(mark, position)
    # composite the watermark with the layer
    return Image.composite(layer, im, layer)

def FacialStamp(frame, Emotion, im_width, im_y, emotion_probability):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img1 = Image.fromarray(frame)
    path = 'EmotionAnalys/utils/Emoji/'

    if Emotion == 'angry':
        mark = Image.open(path + 'angry.png')

    elif Emotion == 'fear':
        mark = Image.open(path + 'fearful.png')

    elif Emotion == 'sad':
        mark = Image.open(path + 'sad.png')

    elif Emotion == 'neutral':
        mark = Image.open(path + 'neutral.png')

    elif Emotion == 'happy':
        mark = Image.open(path + 'happy.png')

    elif Emotion == 'disgust':
        mark = Image.open(path + 'disgust.png')

    elif Emotion == 'surprise':
        mark = Image.open(path + 'surprised.png')

    else:
        mark = Image.open(path + 'NA.png')

    if emotion_probability < 0.5:
        ecolor = (0, 0, 240)

    else:
        ecolor = (0, 240, 0)


    mark.thumbnail((40, 40), Image.ANTIALIAS)
    frame = watermark(img1, mark, (im_width + 5, im_y + 50), 0.6)
    frame = np.array(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.line(frame, (im_width + 5, im_y + 100), (im_width + 80, im_y + 100), (0, 0, 0), 4)
    cv2.line(frame, (im_width + 5, im_y + 100), (im_width + int(emotion_probability * 80), im_y + 100), ecolor, 2)

    return frame


def EmotionRecog(gray_face):
    try:
        gray_face = cv2.resize(gray_face, emotion_target_size)


    except Exception as ex:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        template = "Exception type {0} @EmotionRecog1. Arguments:\n{1!r} in Line {2}"
        message = template.format(type(ex).__name__, ex.args, exc_tb.tb_lineno)
        print(message)
        pass

    gray_face = preprocess_input(gray_face, True)
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)
    emotion_prediction = emotion_classifier.predict(gray_face)
    emotion_probability = np.max(emotion_prediction)
    emotion_label_arg = np.argmax(emotion_prediction)
    emotion_text = emotion_labels[emotion_label_arg]
    emotion_window.append(emotion_text)

    # hyper-parameters for bounding boxes shape
    if len(emotion_window) > 10:
        emotion_window.pop(0)

    try:
        emotion_mode = mode(emotion_window)

    except StatisticsError:
        pass

    except Exception as ex:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        template = "Exception type {0} @EmotionRecog2. Arguments:\n{1!r} in Line {2}"
        message = template.format(type(ex).__name__, ex.args, exc_tb.tb_lineno)
        print(message)
        pass

    return emotion_mode, emotion_probability


while True:

    ret, frame = cap.read()

    if ret is not False:
        gray_image = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
        face_locations = face_recognition.face_locations(frame)

        for i, d in enumerate(face_locations):
            y_dec, w_dec, h_dec, x_dec = d
            face_coordinates = x_dec, y_dec, (w_dec - x_dec), (h_dec - y_dec)
            gray_face = gray_image[y_dec:h_dec, x_dec:w_dec]

            try:
                facial_emotion, emotion_probability = EmotionRecog(gray_face)
                print(facial_emotion, emotion_probability)

                frame = FacialStamp(frame, facial_emotion, w_dec, y_dec, emotion_probability)

            except UnboundLocalError:
                continue

            except Exception as ex:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                template = "Exception type {0} @CallEmotion. Arguments:\n{1!r} in Line {2}"
                message = template.format(type(ex).__name__, ex.args, exc_tb.tb_lineno)
                print(message)
                continue

        cv2.imshow('Facial Emotion', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()