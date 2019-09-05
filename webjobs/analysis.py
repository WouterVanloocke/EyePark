import time
import os
import datetime
import cv2
import pyodbc
from datetime import datetime

#Databankconstanten
server = 'arxusparking.database.windows.net'
database = 'arxusparkingDB'
username = 'stage'
password = 'Wachtwoord123'
driver= '{SQL Server}'
cnxn = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password)
cursor = cnxn.cursor()

uur = datetime.now()
cursor.execute("SELECT parkingid, aantalVrijePlaatsen FROM parkeer_algemeen where status = 1") 
vrijePlaatsen = cursor.fetchall()
for i in vrijePlaatsen:
    cursor.execute("INSERT INTO parkeer_analysis(parkingid, aantalVrijePlaatsen, tijd) VALUES(?, ?, ?)", (i[0], i[1], uur.hour + 2))
cursor.commit()