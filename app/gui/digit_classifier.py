import cv2
import joblib
import numpy as np
import tensorflow as tf
from scipy import ndimage


class DigitClassifier:
    def __init__(self, model):
        self.model_name = model

        if self.model_name == "CNN":
            self.model = tf.keras.models.load_model("assets/trained/cnn_model.h5")
        elif self.model_name == "KNN":
            self.model = joblib.load("assets/trained/knn_model.joblib")

    def predict(self, image):
        image = self.prepareImage(image)
        image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        if self.model_name == "CNN":
            image = image.reshape(1,28,28, 1)
        else:
            image = image.reshape(1, 28*28)

        res = self.model.predict(image)
        return np.argmax(res)

    def prepareImage(self, image):
        image = cv2.bitwise_not(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        while np.sum(gray[0]) == 0:
            gray = gray[1:]

        while np.sum(gray[:, 0]) == 0:
            gray = np.delete(gray, 0, 1)

        while np.sum(gray[-1]) == 0:
            gray = gray[:-1]

        while np.sum(gray[:, -1]) == 0:
            gray = np.delete(gray, -1, 1)

        rows, cols = gray.shape

        if rows > cols:
            factor = 20.0 / rows
            rows = 20
            cols = int(round(cols * factor))
            gray = cv2.resize(gray, (cols, rows))

        else:
            factor = 20.0 / cols
            cols = 20
            rows = int(round(rows * factor))
            gray = cv2.resize(gray, (cols, rows))

        colsPadding = (int(np.math.ceil((28 - cols) / 2.0)), int(np.math.ceil((28 - cols) / 2.0)))
        rowsPadding = (int(np.math.ceil((28 - rows) / 2.0)), int(np.math.ceil((28 - rows) / 2.0)))
        gray = np.lib.pad(gray, (rowsPadding, colsPadding), 'constant')
        
        shiftx, shifty = self.getBestShift(gray)
        shifted = self.shift(gray, shiftx, shifty)
        gray = shifted 
        gray = cv2.resize(gray, (28, 28))
        return gray       
        
    def getBestShift(self, img):
        cy, cx = ndimage.measurements.center_of_mass(img)

        rows, cols = img.shape
        shiftx = np.round(cols / 2.0 - cx).astype(int)
        shifty = np.round(rows / 2.0 - cy).astype(int)

        return shiftx, shifty

    def shift(self, img, sx, sy):
        rows, cols = img.shape
        M = np.float32([[1, 0, sx], [0, 1, sy]])
        shifted = cv2.warpAffine(img, M, (cols, rows))
        return shifted