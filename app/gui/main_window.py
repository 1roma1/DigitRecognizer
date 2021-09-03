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
from PyQt5.QtCore import QTimer

from .paint_widget import PaintWidget


class MainWindow(QMainWindow):
    def __init__(self, connection, prediction):
        QMainWindow.__init__(self)
        self.model_name = "CNN"
        self.setWindowTitle("Digit Recognizer")

        self.centralWidget = QWidget()
        self.centralLayout = QHBoxLayout()

        self.setCentralWidget(self.centralWidget)
        self.centralWidget.setLayout(self.centralLayout)
        self.setUI()

        self.timer = QTimer()
        self.timer.timeout.connect(self.drawText)
        self.timer.start(100)

        self.connection = connection
        self.prediction = prediction

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
        self.model_name = text

    def classify(self):
        image = self.paintWidget.grabImage()
        self.connection.send((image, self.model_name))

    def drawText(self):
        if self.prediction.value >= 0:
            self.numberLabel.setText("Number: " + str(self.prediction.value))
        else:
            self.numberLabel.setText("")