import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic, QtWidgets
import pymysql
import datetime
import Boundary_Main as bm
form_login = uic.loadUiType("login.ui")[0]


class Boundary_Login(QMainWindow,form_login):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.open_main)

    def open_main(self):
        if self.lineEdit.text() == 'manager' and self.lineEdit_2.text() == '1234':
            print("로그인 성공")
            self.close()
            a = bm.Boundary_Main(self)
        else:
            buttonReply = QMessageBox.warning(
 
            self, '경고', "로그인 정보가 일치하지 않습니다.",
 
            QMessageBox.Yes)


app = QApplication(sys.argv)
mainWindow = Boundary_Login()
mainWindow.show()
app.exec_()