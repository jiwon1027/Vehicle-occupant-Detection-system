class illegal_parking_information:
    carnumber=""
    location=""
    fine=0.0 #double 
    overlap=0 #int
    date="" #string
    picture=""#BLOB 사진은 일단 string으로 해놧음 ...
    def __init__(self):
        self.carnumber=""
        self.location=""
        self.fine=0.0
        self.overlap=0
        self.date=""

    def input_data(self,carnumber,location,date,fine,picture): #picture
        self.carnumber=carnumber
        self.location=location
        self.date=date
        self.fine=fine
        self.overlap=0
        self.picture=picture
    def getcarnumber(self): #carnumber 반환
        return self.carnumber
    def output_data(self): #list 반환
        return self  
    def setoverlap(self,num): #같은 차량이 여러번 불법주차할수있기때문에 그에대한 횟수값
        self.overlap=num