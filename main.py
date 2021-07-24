from PyQt5.QtWidgets import *
from app.identify_digit import IdentifyDigit

if __name__ == '__main__':
    import sys 
    app = QApplication(sys.argv)
    ex = IdentifyDigit()
    ex.resize(800, 600)
    ex.show()
    sys.exit(app.exec_())