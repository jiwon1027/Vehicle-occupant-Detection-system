import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic, QtWidgets
import Boundary_Action as ba
import Boundary_parking as bp
import Boundary_Fine as bf

class Boundary_Main(QDialog):
    def __init__(self, parent):
        super(Boundary_Main, self).__init__(parent)
        uic.loadUi("Main_ui.ui", self)
        self.show()
        self.pushButton.clicked.connect(self.open_parking)
        self.pushButton_2.clicked.connect(self.open_action)
        self.pushButton_3.clicked.connect(self.open_fine)

    def open_parking(self):
        a = bp.Boundary_Parking(self)

    def open_action(self):
        b = ba.Boundary_Action(self)

    def open_fine(self):
        c = bf.Boundary_Fine(self)
