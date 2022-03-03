import pymysql
import Entity_illegal

class Control_fine:
    def __init__(self, carnum, add):
        self.__num = carnum
        self.__ad = add


    def update(self):
        connect3 = pymysql.connect(host='113.198.234.39', user='root', password='111111',

                       db='project', charset='utf8')
        cur = connect3.cursor()
        sql = "Update illegal_carnumber SET fine='"+str(self.__ad)+"' WHERE carnumber='"+str(self.__num)+"'"
        cur.execute(sql)
        connect3.commit()
        connect3.close()