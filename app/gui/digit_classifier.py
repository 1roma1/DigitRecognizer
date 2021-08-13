import os
import cv2
import numpy as np
import tensorflow as tf
from scipy import ndimage

class DigitClassifier:
    def __init__(self):
        self.model = tf.keras.models.load_model("../models/model.h5")

    def predict(self, image):
        image = self.prepareImage(image)
        image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        #Reshapes image
        image = image.reshape(1,28,28, 1)       
        
        #predicting the class
        res = self.model.predict([image])[0]
        return np.argmax(res), max(res)

    def prepareImage(self, image):
        #Invert image to get whie digit on black background
        image = cv2.bitwise_not(image)

        #Converts image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #resizes image to be 28x28
        (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
        #First we want to fit the images into this 20x20 pixel box. 
        #Therefore we need to remove every row and column at the sides of the image 
        #which are completely black.
        while np.sum(gray[0]) == 0:
            gray = gray[1:]

        while np.sum(gray[:, 0]) == 0:
            gray = np.delete(gray, 0, 1)

        while np.sum(gray[-1]) == 0:
            gray = gray[:-1]

        while np.sum(gray[:, -1]) == 0:
            gray = np.delete(gray, -1, 1)

        rows, cols = gray.shape

        #Now we want to resize our outer box to fit it into a 20x20 box. 
        #We need a resize factor for this.
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

        #at the end we need a 28x28 pixel image so we add the missing black rows and columns 
        #using the np.lib.pad function which adds 0s to the sides.
        colsPadding = (int(np.math.ceil((28 - cols) / 2.0)), int(np.math.ceil((28 - cols) / 2.0)))
        rowsPadding = (int(np.math.ceil((28 - rows) / 2.0)), int(np.math.ceil((28 - rows) / 2.0)))
        gray = np.lib.pad(gray, (rowsPadding, colsPadding), 'constant')
        #gray = np.lib.pad(gray, 2, 'constant',constant_values=255)
        
        shiftx, shifty = self.getBestShift(gray)
        shifted = self.shift(gray, shiftx, shifty)
        gray = shifted 
        gray = cv2.resize(gray, (28, 28))
        return gray       
        
    def getBestShift(self,img):
        cy, cx = ndimage.measurements.center_of_mass(img)

        rows, cols = img.shape
        shiftx = np.round(cols / 2.0 - cx).astype(int)
        shifty = np.round(rows / 2.0 - cy).astype(int)

        return shiftx, shifty

    def shift(self,img, sx, sy):
        rows, cols = img.shape
        M = np.float32([[1, 0, sx], [0, 1, sy]])
        shifted = cv2.warpAffine(img, M, (cols, rows))
        return shifted