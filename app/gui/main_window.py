from PyQt5.QtWidgets import (
    QMainWindow, 
    QWidget, 
    QHBoxLayout, 
    QVBoxLayout,
    QPushButton, 
    QLabel
)
from PyQt5.QtGui import QFont

from gui.paint_widget import PaintWidget
from gui.digit_classifier import DigitClassifier

class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle("Digit Recognizer")

        self.centralWidget = QWidget()
        self.centralLayout = QHBoxLayout()

        self.setCentralWidget(self.centralWidget)
        self.centralWidget.setLayout(self.centralLayout)
        self.setUI()

        self.classifier = DigitClassifier()

    def setUI(self):
        self.paintWidgetLayout = QVBoxLayout()
        self.paintWidget = PaintWidget(self)
        self.clearButton = QPushButton("Clear")
        self.paintWidgetLayout.addWidget(self.paintWidget)
        self.paintWidgetLayout.addWidget(self.clearButton)
        self.clearButton.clicked.connect(self.clearCanvas)

        self.predictionLayout= QVBoxLayout()
        self.numberLabel = QLabel()
        self.accuracyLabel = QLabel()
        self.numberLabel.setFont(QFont("Times", 14))
        self.accuracyLabel.setFont(QFont("Times", 14))
        self.predictButton = QPushButton("Predict")
        self.predictionLayout.addWidget(self.numberLabel)
        self.predictionLayout.addWidget(self.accuracyLabel)
        self.predictionLayout.addWidget(self.predictButton)
        self.predictButton.clicked.connect(self.classify)
        
        self.centralLayout.addLayout(self.paintWidgetLayout)
        self.centralLayout.addLayout(self.predictionLayout)

    def  clearCanvas(self):
        self.paintWidget.clear()
        self.paintWidget.show()

    def classify(self):
        image = self.paintWidget.grabImage()        
        digit, acc = self.classifier.predict(image)

        self.numberLabel.setText("Number: " + str(digit))
        self.accuracyLabel.setText("Accuracy: " + str(int(acc*100)))