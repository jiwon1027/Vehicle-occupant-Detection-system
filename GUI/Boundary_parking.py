import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic, QtWidgets
import pymysql
import datetime
import Boundary_insert as bi
import Control_parking as cp
import pandas as pd
from sqlalchemy import create_engine
from PIL import Image
import base64
from io import BytesIO
from PyQt5.QtCore import pyqtSlot, Qt

class Boundary_Parking(QDialog):
    def __init__(self, parent):
        super(Boundary_Parking, self).__init__(parent)
        uic.loadUi("parking_ui.ui", self)
        self.data_show()
        
        self.pushButton.clicked.connect(self.data_search)
        self.pushButton_2.clicked.connect(self.data_delete)
        self.pushButton_3.clicked.connect(self.data_add)
        self.pushButton_4.clicked.connect(self.data_update)
        self.pushButton_5.clicked.connect(self.data_show)
        self.tableWidget.doubleClicked.connect(self.pic_show)
        self.show()

    def pic_show(self):
        try:
            engine = create_engine('mysql+pymysql://root:111111@113.198.234.49/test', echo = False)
            img_read1 = pd.read_sql(sql='select pic from test_ where carnumber = '+ "'" +str(self.tableWidget.item(int(self.tableWidget.currentIndex().row()), 1).text())+"'", con=engine)
            img_str2 = img_read1['pic'].values[0]
            img = base64.decodestring(img_str2)
            im = Image.open(BytesIO(img))
            im.show()
        except:
            print("입력된 사진이 없습니다.")
        
        
        

    def data_delete(self):
        if self.lineEdit.text():
            var = cp.Control_delete()
            var.setCar(self.lineEdit.text())
            var.delete()
            self.data_show()
        else:
            buttonReply = QMessageBox.information(
                    self, '검색 정보', "정보를 입력해주세요.",
                    QMessageBox.Yes)


    def data_search(self):
        a = ""
        if self.lineEdit.text():
            for i in range(int(self.tableWidget.rowCount())):
                if str(self.tableWidget.item(i,1).text()) == str(self.lineEdit.text()):
                    self.label_5.setText(str(self.tableWidget.item(i,3).text())+"원")
                    self.label_5.repaint()
                    self.label_3.setText(str(self.tableWidget.item(i,1).text()))
                    self.label_3.repaint()


                    a = a +"날짜 : " +str(self.tableWidget.item(i,0).text())
                    a = a +", 차량번호 :" +str(self.tableWidget.item(i,1).text())
                    a = a +", 사람 수 : " +str(self.tableWidget.item(i,2).text())
                    a = a +", 벌금 : " +str(self.tableWidget.item(i,3).text())

                    break
            if a!="":
                buttonReply = QMessageBox.information(
                    self, '검색 정보', a,
                    QMessageBox.Yes)
            else:
                buttonReply = QMessageBox.information(
                    self, '검색 정보', "일치하는 정보가 없습니다.",
                    QMessageBox.Yes)
                self.lineEdit.setText("")
                self.lineEdit.repaint()            
        else:
            buttonReply = QMessageBox.warning(
 
            self, '경고', '정보를 입력해주세요',
            QMessageBox.Yes)
            
    

    def data_add(self):
        bi.Boundary_insert(self)
        

    def data_show(self):
        connect1 = pymysql.connect(host='113.198.234.49', user='root', password='111111',
                       db='test', charset='utf8')
        cur = connect1.cursor()
        sql = "select * from test_"
        cur.execute(sql)
        row = cur.fetchall()
        connect1.close()
        column_headers =  ['날짜', '차량번호','사람수','벌금']
        self.tableWidget.setColumnCount(len(column_headers))
        self.tableWidget.setRowCount(row.__len__())
        self.tableWidget.setHorizontalHeaderLabels(column_headers)

        for i in range(int(self.tableWidget.rowCount())):
            for j in range(int(self.tableWidget.columnCount())):
                self.tableWidget.setItem(i,0,QTableWidgetItem(row[i][0]))
                self.tableWidget.setItem(i,1,QTableWidgetItem(row[i][1]))
                self.tableWidget.setItem(i,2,QTableWidgetItem(str(row[i][2])))

                item1 = QTableWidgetItem()
                item1.setData(Qt.DisplayRole, row[i][3])

                self.tableWidget.setItem(i,3,item1)



        self.tableWidget.resizeColumnsToContents()
        self.tableWidget.resizeRowsToContents()
        

    def data_update(self):
        a = cp.Control_update(self.lineEdit_2.text(), self.label_3.text())
        if self.label_5.text()=="":
            buttonReply = QMessageBox.warning(
            self, '경고', "존재하지 않는 정보입니다.",
            QMessageBox.Yes)
            self.lineEdit.setText("")
            self.lineEdit.repaint()
            self.lineEdit_2.setText("")
            self.lineEdit_2.repaint()
        else:
            a.data_update()
            self.label_3.setText("")
            self.label_3.repaint()
            self.label_5.setText("")
            self.label_5.repaint()
            self.lineEdit.setText("")
            self.lineEdit.repaint()
            self.lineEdit_2.setText("")
            self.lineEdit_2.repaint()
            buttonReply = QMessageBox.information(
            self, '성공', "정상적으로 정보가 업데이트 되었습니다.",
            QMessageBox.Yes)
            self.data_show()