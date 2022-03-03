import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic, QtWidgets
import pymysql
import datetime
import Control_fine as cf
import Boundary_Main as bm
from PyQt5.QtCore import pyqtSlot, Qt




class Boundary_Fine(QDialog):
    def __init__(self,parent):
        super(Boundary_Fine, self).__init__(parent)
        uic.loadUi("Fine_ui.ui", self)
        self.data_show()
        

        self.pushButton_2.clicked.connect(self.update)
        self.pushButton.clicked.connect(self.search)
        
        self.show()

    def data_show(self):
        connect2 = pymysql.connect(host='113.198.234.49', user='root', password='111111',
                       db='test', charset='utf8')

        cur = connect2.cursor()
        sql = "select * from test_"
        cur.execute(sql)
        row = cur.fetchall()
        column_headers =  ['날짜', '차량번호','사람수','벌금','위반횟수']
        self.tableWidget.setColumnCount(len(column_headers))
        self.tableWidget.setRowCount(row.__len__())
        self.tableWidget.setHorizontalHeaderLabels(column_headers)



        for i in range(int(self.tableWidget.rowCount())):
            for j in range(int(self.tableWidget.columnCount())):
                self.tableWidget.setItem(i,0,QTableWidgetItem(row[i][0]))
                self.tableWidget.setItem(i,1,QTableWidgetItem(row[i][1]))
                self.tableWidget.setItem(i,2,QTableWidgetItem(row[i][2]))

                item1 = QTableWidgetItem()
                item2 = QTableWidgetItem()
                item1.setData(Qt.DisplayRole, row[i][3])
                item2.setData(Qt.DisplayRole, row[i][4])

                self.tableWidget.setItem(i,3,item1)
                self.tableWidget.setItem(i,4,item2)

        connect2.close()

    def update(self):
        if self.label_5.text()=="":
            buttonReply = QMessageBox.warning(
            self, '경고', "존재하지 않는 정보입니다.",
 
            QMessageBox.Yes)
            self.lineEdit.setText("")
            self.lineEdit.repaint()
            self.lineEdit_2.setText("")
            self.lineEdit_2.repaint()
        else:
            add=self.won+int(self.lineEdit_2.text())
            print(add)
            a = cf.Control_fine(self.carnumber,add)
            a.update()
            self.label_5.setText("")
            self.label_5.repaint()
            self.label_3.setText("")
            self.label_3.repaint()
            self.lineEdit_2.setText("")
            self.lineEdit_2.repaint()
            buttonReply = QMessageBox.information(
            self, '성공', "정상적으로 벌금이 부과되었습니다.",
            QMessageBox.Yes)

            self.data_show()
      
        

    def search(self):
        connect4 = pymysql.connect(host='113.198.234.49', user='root', password='111111',

                       db='test', charset='utf8')

        self.carnumber=self.lineEdit.text()
        cur = connect4.cursor()
        sql = "Select carnumber,fine from test_ where carnumber='"+self.carnumber+"'"
        cur.execute(sql)
        self.result=cur.fetchone()
        print(self.result)
        if self.result == None:
            buttonReply = QMessageBox.warning(
            self, '경고', "존재하지 않는 정보입니다.",
 
            QMessageBox.Yes)
            self.lineEdit.setText("")
            self.lineEdit.repaint()
            self.lineEdit_2.setText("")
            self.lineEdit_2.repaint()
        else:
            self.carnumber=str(self.result[0])
            self.won=int(self.result[1])
            self.result=cur.fetchone()
            self.label_3.setText(str(self.won)+"원")
            self.label_3.repaint()
            self.label_5.setText(self.carnumber)
            self.label_5.repaint()
            self.lineEdit.setText("")
            self.lineEdit.repaint()
        connect4.close()