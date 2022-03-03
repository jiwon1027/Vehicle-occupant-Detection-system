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

class parking_obstruction_information:
    date=""
    picture=""    
    def __init__(self):
        self.data=""
        self.picture=""
    def setpicture(self,picture):
        self.picture=picture
    def setdate(self,date):
        self.date=date
    def getdate(self):
        return self.date
    def getpicture(self,picture):
        return self.picture

