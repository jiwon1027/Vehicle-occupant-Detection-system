import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic, QtWidgets
import datetime
import Control_parking as cp

class Boundary_insert(QDialog):
    def __init__(self, parent):
        super(Boundary_insert, self).__init__(parent)
        uic.loadUi("Boundary_insert.ui", self)   
        self.pushButton.clicked.connect(self.car_insert)
        self.show()
    
    def car_insert(self):
        b = str(self.lineEdit_2.text()).isalnum()
        if b == True and (len(str(self.lineEdit_2.text())) == 7 or len(str(self.lineEdit_2.text())) == 8):
            a = cp.Control_insert(self.lineEdit.text(),self.lineEdit_2.text(),self.lineEdit_3.text(),self.lineEdit_4.text())
            a.insert()
        else:
            buttonReply = QMessageBox.warning(
            self, '경고', "제대로 입력해 주세요.",
 
            QMessageBox.Yes)



        self.close()
        
        