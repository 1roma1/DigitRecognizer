import sys
import multiprocessing as mp
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow
from recognition.digit_classifier import model_process

if __name__ == '__main__':
    parent, child = mp.Pipe()
    prediction = mp.Value('i', -1)

    p = mp.Process(target=model_process, args=(child, prediction))
    p.start()
    parent.recv()

    app = QApplication(sys.argv)
    ex = MainWindow(parent, prediction)
    ex.resize(800, 600)
    ex.show()
    app.exec_()
    p.kill()
