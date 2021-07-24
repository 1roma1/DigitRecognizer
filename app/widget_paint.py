from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import cv2
import numpy as np

class WidgetPaint(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)

        self.setGeometry(20, 20, 400, 400)
        self.image = QImage(self.size(), QImage.Format_RGB32)  
        self.image.fill(Qt.white)
        self.drawing = False
        self.brushSize = 40
        self.brushColor = Qt.black
        self.lastPoint = QPoint()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()
        
    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) & self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(self.brushColor, self.brushSize, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()
        
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
    
    def paintEvent(self, event):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())

    def return_image(self):
        cv2.imwrite('digit.png', self.QImageToCvMat(self.image))

    def clear(self):
        self.image.fill(Qt.white)
        self.update()
    
    def save(self):
        self.image.save("digit.png")

    def QImageToCvMat(self,incomingImage):
        '''  Converts a QImage into an opencv MAT format  '''

        incomingImage = \
           incomingImage.convertToFormat(QImage.Format.Format_RGBA8888)

        width = incomingImage.width()
        height = incomingImage.height()

        ptr = incomingImage.bits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        return arr