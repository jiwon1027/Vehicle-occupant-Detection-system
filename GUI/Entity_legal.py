class legal_car_information:
    car_type=""
    carnumber=""
    def __init__(self):
        self.car_type=""
        self.carnumber=""
    
    def setcarnumber(self,carnumber):
        self.carnumber=carnumber
    def setcar_type(self,car_type):
        self.car_type=car_type
    def getcarnumber(self):
        return self.carnumber
    def getcar_type(self):
        return self.car_type