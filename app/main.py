from PyQt5.QtWidgets import QApplication
from app.gui.main_window import MainWindow

if __name__ == '__main__':
    import sys 
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.resize(800, 600)
    ex.show()
    sys.exit(app.exec_())