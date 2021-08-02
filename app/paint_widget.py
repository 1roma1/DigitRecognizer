import numpy as np

from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage, QPainter, QPen
from PyQt5.QtCore import QPoint, Qt

class PaintWidget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)

        self.setGeometry(20, 20, parent.size().width()/2, parent.size().height()/2)
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

    def resizeEvent(self, event):
        self.image = self.image.scaled(self.size(), Qt.AspectRatioMode.IgnoreAspectRatio)
        return super().resizeEvent(event)

    def grabImage(self):
        image = self.image.convertToFormat(QImage.Format.Format_RGBA8888)

        width = image.width()
        height = image.height()

        ptr = image.bits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        return arr

    def clear(self):
        self.image.fill(Qt.white)
        self.update()