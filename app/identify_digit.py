
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from app.widget_paint import WidgetPaint
from PIL import ImageGrab, Image
import numpy as np
from tensorflow.keras.models import load_model
import cv2
from scipy import ndimage

class IdentifyDigit(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle("Predicting Digits with CNN")

        self.centralWidget = QWidget()
        self.layout = QHBoxLayout()

        self.setCentralWidget(self.centralWidget)
        self.centralWidget.setLayout(self.layout)
        self.setUI()

        self.model = load_model('model/model.h5')

    def setUI(self):
        self.layout1 = QVBoxLayout()
        self.wp = WidgetPaint()
        self.layout1.addWidget(self.wp)

        self.layout11 = QHBoxLayout()
        self.saveButton = QPushButton("Save")
        self.clearButton = QPushButton("Clear")
        self.layout11.addWidget(self.saveButton)
        self.layout11.addWidget(self.clearButton)
        self.clearButton.clicked.connect(self.clear_canvas)
        self.saveButton.clicked.connect(self.save_canvas)
        

        self.layout1.addLayout(self.layout11)

        self.layout2 = QVBoxLayout()
        self.label1 = QLabel("Label1")
        self.label2 = QLabel("Label2")
        self.predictButton = QPushButton("Predict")
        self.layout2.addWidget(self.label1)
        self.layout2.addWidget(self.label2)
        self.layout2.addWidget(self.predictButton)
        self.predictButton.clicked.connect(self.classify_handwriting)
        
        self.layout.addLayout(self.layout1)
        self.layout.addLayout(self.layout2)

    def  clear_canvas(self):
        self.wp.clear()
        self.wp.show()
    
    def save_canvas(self):
        self.wp.save()

    def predict_digit(self, img):  
        #Normalize image to range [0 1]
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        #Reshapes image
        img = img.reshape(1,28,28,1)       
        
        #predicting the class
        res = self.model.predict([img])[0]
        return np.argmax(res), max(res)

    def classify_handwriting(self):
        self.wp.return_image()        
        #im = Image.open("test.png")
        
        im = self.imageprepare("digit.png")
        digit, acc = self.predict_digit(im)
        self.label1.setText('Predicted= '+str(digit))
        self.label2.setText('Accuracy= '+ str(int(acc*100)))

    def imageprepare(self,fileName):       
        image = cv2.imread(fileName)

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
        cv2.imwrite("result.png", gray)
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

