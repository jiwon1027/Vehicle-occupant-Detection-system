import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic, QtWidgets
import pymysql
import datetime
import Entity_legal
import Entity_illegal

class Control_parking: #컨트롤 부모클래스
    carnumber=""
    date=""
    fine=0.0
    picture=""
    location=""
    overlap =0
    def __init__(self):
        print("hi")
    def __init__(self,carnumber):
        self.carnumber=carnumber
    def search(self,carnumber): 
        print("bye")
        #입력받은 차넘버로 조회하여 존재하면 true 없으면 false
        #여기서 select문을 써서 있으면 true 없으면 false 로 boolea값보냄



class Control_delete(Control_parking):
    def __init__(self):
        self.__car=""
        #print("ㅇㅁㄴㅇㄴㅁ")
    def setCar(self, carnum):
        self.__car=carnum

    def delete(self):
        connect1 = pymysql.connect(host='113.198.234.49', user='root', password='111111',
                       db='test', charset='utf8')

        num=0 
        if self.__car != "":
            cur = connect1.cursor()
            sql = "select * from test_ where carnumber = '" + self.__car + "'"
            cur.execute(sql)
            row = cur.fetchone()
            print(row)
            print(type(row))
            if row:
                cur = connect1.cursor()
                sql = "delete from test_ where carnumber = '" + self.__car + "'"
                cur.execute(sql)
                connect1.commit()              
        else:
            buttonReply = QMessageBox.warning( 
            self, '경고', '정보를 입력해주세요',
            QMessageBox.Yes)
                
        connect1.close()

class Control_insert(Control_parking):
    def __init__(self, date, carnumber,location, fine):
        self.__num = carnumber
        self.__fine = fine
        self.__datet = date
        self.__overlap = 0
        self.__location = location

    def getNum(self):
        return self.__num
    def getFine(self):
        return self.__fine
    def getDate(self):
        return self.__datet
    def getlocation(self):
        return self.__location

    def setDate(self, var):
        self.__datet = var
    
    def insert(self):
        connect = pymysql.connect(host='113.198.234.49', user='root', password='111111',

                       db='test', charset='utf8')
        cur = connect.cursor()
        sql = "select * from test_ where carnumber = '" + self.__num + "'"
        cur.execute(sql)
        row = cur.fetchone()
        
        if not row:
            sql = "insert into test_ values('" + self.__datet +"', '" + self.__num  + "','" + self.__location + "', '"+ self.__fine + "', 0 , NULL )"
            cur.execute(sql)           
        else:
            print("중복")
            total = int(row[4]) + 1
            money = int(row[3]) + int(self.__fine)
            print(total)
            sql = "update test_ set overlap ='"+ str(total) + "'where carnumber='"+ self.__num +"'"
            cur.execute(sql)  
            sql = "update test_ set fine ='"+ str(money) + "'where carnumber='"+ self.__num +"'"
            cur.execute(sql) 
        connect.commit()
        connect.close()
        

class Control_update(Control_parking):
    def __init__(self, text1, text2):
        self.__line_text1 = text1
        self.__line_text2 = text2

    def data_update(self):
        connect = pymysql.connect(host='113.198.234.49', user='root', password='111111',

                       db='test', charset='utf8')

        cur = connect .cursor()
        sql = "Update test_ SET carnumber='"+self.__line_text1+"' WHERE carnumber='"+self.__line_text2+"'"
        cur.execute(sql)
        connect.commit()
        connect.close()
        