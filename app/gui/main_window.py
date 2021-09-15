from threading import Thread
from PyQt5.QtWidgets import (
    QMainWindow, 
    QWidget, 
    QHBoxLayout, 
    QVBoxLayout,
    QPushButton, 
    QLabel,
    QComboBox,
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot

from .paint_widget import PaintWidget
from recognition.digit_classifier import DigitClassifier

class PredictionThread(QThread):

    predict = pyqtSignal(str)

    def __init__(self, clf, parent=None):
        QThread.__init__(self, parent)
        self.clf = clf

    def setData(self, img, modelType):
        self.img = img
        self.modelType = modelType

    def run(self):
        prediction = self.clf.predict(self.img, self.modelType)
        self.predict.emit(str(prediction))


class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.modelType = "CNN"
        self.setWindowTitle("Digit Recognizer")

        self.centralWidget = QWidget()
        self.centralLayout = QHBoxLayout()

        self.setCentralWidget(self.centralWidget)
        self.centralWidget.setLayout(self.centralLayout)
        self.setUI()

        self.clf = DigitClassifier()
        self.predictionThread = PredictionThread(self.clf)


    def setUI(self):
        self.paintWidgetLayout = QVBoxLayout()

        self.paintWidget = PaintWidget(self)
        self.clearButton = QPushButton("Clear")

        self.paintWidgetLayout.addWidget(self.paintWidget)
        self.paintWidgetLayout.addWidget(self.clearButton)

        self.clearButton.clicked.connect(self.clearCanvas)

        self.predictionLayout= QVBoxLayout()

        self.numberLabel = QLabel()
        self.numberLabel.setFont(QFont("Times", 14))
        self.modelSelector = QComboBox()
        self.modelSelector.addItems(["CNN", "KNN"])
        self.predictButton = QPushButton("Predict")

        self.predictionLayout.addWidget(self.numberLabel)
        self.predictionLayout.addWidget(self.modelSelector)
        self.predictionLayout.addWidget(self.predictButton)

        self.predictButton.clicked.connect(self.classify)
        self.modelSelector.activated[str].connect(self.selectModel)
        
        self.centralLayout.addLayout(self.paintWidgetLayout)
        self.centralLayout.addLayout(self.predictionLayout)

    def  clearCanvas(self):
        self.paintWidget.clear()
        self.paintWidget.show()

    def selectModel(self, text):
        self.modelType = text

    def classify(self):
        image = self.paintWidget.grabImage()
        self.predictionThread.predict.connect(self.setPredict)
        self.predictionThread.setData(image, self.modelType)
        self.predictionThread.start()

    @pyqtSlot(str)
    def setPredict(self, prediction):
        self.numberLabel.setText("Number: " + prediction)