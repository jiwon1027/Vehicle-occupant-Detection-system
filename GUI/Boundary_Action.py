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

class Boundary_Action(QDialog):
    def __init__(self, parent):
        super(Boundary_Action, self).__init__(parent)
        uic.loadUi("Action_ui.ui", self)
        self.data_show()
        self.tableWidget.doubleClicked.connect(self.pic_show)
        self.show()

    def data_show(self):
        connect1 = pymysql.connect(host='113.198.234.49', user='root', password='111111',
                       db='test', charset='utf8')
        cur = connect1.cursor()
        sql = "select * from test_"
        cur.execute(sql)
        row = cur.fetchall()
        connect1.close()
        column_headers = ['날짜', '번호판', '사람 수']
        self.tableWidget.setColumnCount(len(column_headers))
        self.tableWidget.setRowCount(row.__len__())
        self.tableWidget.setHorizontalHeaderLabels(column_headers)

        for i in range(int(self.tableWidget.rowCount())):
            for j in range(int(self.tableWidget.columnCount())):
                self.tableWidget.setItem(i,j,QTableWidgetItem(row[i][j]))

        self.tableWidget.resizeColumnsToContents()
        self.tableWidget.resizeRowsToContents()

    def pic_show(self):
        engine = create_engine('mysql+pymysql://root:111111@113.198.234.49/test', echo=False)
        img_read1 = pd.read_sql(sql='select pic from test_ where date = '+ "'" +str(self.tableWidget.item(int(self.tableWidget.currentIndex().row()), 0).text())+"'", con=engine)
        img_str2 = img_read1['pic'].values[0]
        img = base64.decodestring(img_str2)
        im = Image.open(BytesIO(img))
        im.show()