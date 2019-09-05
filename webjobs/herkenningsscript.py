from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateEntry, Region
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from datetime import *
import time
import os
import datetime
import cv2
import pyodbc
from pathlib import Path
import stat

#Databankconstanten
server = 'arxusparking.database.windows.net'
database = 'arxusparkingDB'
username = 'stage'
password = 'Wachtwoord123'
driver= '{SQL Server}'
cnxn = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password)
cursor = cnxn.cursor()

ENDPOINT = "https://westeurope.api.cognitive.microsoft.com"
# Plaats hier de keys van je subscription
training_key = "f0dda4df7d7d4cf7a929e471a8aacbe3"
prediction_key = "085fb3c741924d09927420ad56be35d4"
prediction_resource_id = "/subscriptions/3d325e01-eb54-447a-b415-75dc9f70c03f/resourceGroups/customVision/providers/Microsoft.CognitiveServices/accounts/customVision_prediction"

# Plaats hier de juiste naam van je iteratie
publish_iteration_name = "Iteration1"

trainer = CustomVisionTrainingClient(training_key, endpoint=ENDPOINT)

# Zoek het object detection domain
obj_detection_domain = next(domain for domain in trainer.get_domains() if domain.type == "ObjectDetection" and domain.name == "General")

# Zet hier de key van je project
project = trainer.get_project("97181c71-c821-4f8f-8571-3d7ff56ff334")

#########################Met deze code kan je het model trainen#####################################
#iteration = trainer.get_iteration
#iteration = trainer.train_project(project.id)
#while (iteration.status != "Completed"):
#    iteration = trainer.get_iteration(project.id, iteration.id)
#    print ("Training status: " + iteration.status)
#    time.sleep(1)
#trainer.publish_iteration(project.id, iteration.id, publish_iteration_name, prediction_resource_id)

predictor = CustomVisionPredictionClient(prediction_key, endpoint=ENDPOINT)

#Initialiseren van variabelen
nu = datetime.datetime.now()
today = date.today()

#path to files
local_path = Path("D:/home/foto")

for i in os.listdir(local_path):
    for subdir, dirs, files in os.walk(local_path / i):
        for file in files:
            if file.endswith('.jpg'):
                cursor.execute("SELECT maxbereikt from parkeer_algemeen where parkingid = ?", i)
                maxbereikt = cursor.fetchone()[0]
                if(maxbereikt == "Ja"):
                    break
                else:
                    foto = os.path.join(subdir, file)

                    font = cv2.FONT_HERSHEY_DUPLEX
                    video_capture = cv2.VideoCapture(foto)
                    width = video_capture.get(3)
                    height = video_capture.get(4)

                    countbezet = 0
                    idlist = []
                    listje = []
                    coordinaten = []
                    bezetteplaatsen = []
                    vrijeplaatsenlijst = []
                    coordinatendict = {}

                    cursor.execute("SELECT maxaantalplaatsen from parkeer_algemeen where parkingid = ?", i)
                    maxaantalplaatsen = cursor.fetchone()[0]
                    cursor.execute("SELECT xcoördinaat,ycoördinaat from parkeer_plaatstemp where parkingid = ?", i)
                    tijdig = cursor.fetchall()
         
                    while video_capture.isOpened():
                        success, frame = video_capture.read()
                        if not success:
                            break

                        with open(foto, mode="rb") as test_data:
                            results = predictor.detect_image(project.id, publish_iteration_name, test_data)

                        # Resultaten laten zien
                        for prediction in results.predictions:
                            links = int(prediction.bounding_box.left * width)
                            boven = int(prediction.bounding_box.top * height)
                            rechts = int((prediction.bounding_box.width * width) + (prediction.bounding_box.left * width))
                            beneden = int((prediction.bounding_box.height * height) + (prediction.bounding_box.top * height))

                            #Het model heeft een lege plaats gezien.
                            if prediction.tag_name == "auto" and prediction.probability * 100 > 60:
                                #Coördinaten van middelpunt van rechthoek berekenen
                                x = links + ((rechts - links)/2)
                                y = boven + ((beneden - boven)/2)
                                cursor.execute("SELECT COUNT (*) from parkeer_exacteplaats where parkingid = ?", i)
                                countplaatsen = cursor.fetchone()[0]
                                cursor.execute("SELECT xcoördinaat, ycoördinaat FROM parkeer_exacteplaats where parkingid = ?", i)
                                controle = cursor.fetchall()

                                if not tijdig:
                                    cursor.execute("INSERT INTO parkeer_plaatstemp (xcoördinaat,ycoördinaat,parkingid) VALUES (?, ?, ?)", x,y,i)
                                    print("tijdelijke plaats toegevoegd")
                                    cursor.commit()
                                else:
                                    for plaats in tijdig:
                                        if not controle:
                                            #plaats toevoegen wanneer het middelpunt binnen de 70 pixels ligt van de waarden in de databank
                                            if plaats[0]-70 <= x <= plaats[0]+70 and plaats[1]-70 <= y <= plaats[1]+70:
                                                print("exacteplaats toegevoegd zonder dat er iets in de databank zat.")
                                                cursor.execute("INSERT INTO parkeer_exacteplaats (xcoördinaat,ycoördinaat,parkingid) VALUES (?, ?, ?)", x,y,i)
                                                cursor.commit()
                                        else:
                                            if plaats[0]-70 <= x <= plaats[0]+70 and plaats[1]-70 <= y <= plaats[1]+70:
                                                cursor.execute("SELECT xcoördinaat,ycoördinaat from parkeer_exacteplaats where xcoördinaat between ? and ? and ycoördinaat between ? and ? and parkingid = ?", x-70, x+70, y-70, y+70, i)
                                                dubbel = cursor.fetchall()
                                                print('Dubbel = ' + str(dubbel))
                                                if not dubbel:
                                                    cursor.execute("INSERT INTO parkeer_exacteplaats (xcoördinaat,ycoördinaat,parkingid) VALUES (?, ?, ?)", x,y,i)
                                                    print("exacte plaats toegevoegd")
                                                    #counter += 1
                                                    cursor.commit()
                                
                                    cursor.execute("DELETE FROM parkeer_plaatstemp WHERE parkingid = ?",i)
                                    cursor.commit()
                                    
                                cursor.execute("SELECT id, xcoördinaat,ycoördinaat from parkeer_exacteplaats where parkingid = ?", i)
                                vergelijkplaats = cursor.fetchall()
                                for z in vergelijkplaats:  
                                    idlist.append(z[0])
                                    if z[1]-70 <= x <= z[1]+70 and z[2]-70 <= y <= z[2]+70:
                                        countbezet += 1
                                        cv2.circle(frame, (int(x), int(y)), 15, (0, 0, 255), 3)
                                        listje.append(z[0])

                        vrijeplaatsenlijst = list(set(idlist) - set(listje))

                        cursor.execute("SELECT maxaantalplaatsen from parkeer_algemeen where parkingid = ?", i)
                        aantalplaatsen = cursor.fetchone()[0]
                        

                        
                        for lijn in vrijeplaatsenlijst:
                            cursor.execute("SELECT xcoördinaat,ycoördinaat from parkeer_exacteplaats where id = ?", int(lijn))
                            coordinaten = cursor.fetchall()
                            coordinatendict.update(coordinaten)
                    
                        for x,y in coordinatendict.items():
                            cv2.circle(frame, (x, y), 15, (0, 255, 0), 3)

                        vrijeplaatsen = len(vrijeplaatsenlijst)

                        if countbezet > maxaantalplaatsen:
                            countbezet = maxaantalplaatsen
                        
                        #Foto wegschrijven zodat deze gebruikt kan worden door de website
                        cv2.imwrite('D:/home/site/wwwroot/static/images/fotometrechthoeken' + i + '.jpg', frame)
                        print('foto ' + i + ' wegschrijven')

                        #Maximum aantal plaatsen in de databank aanpassen
                        cursor.execute("SELECT COUNT (*) from parkeer_exacteplaats where parkingid = ?", i)
                        maxplaatsen = cursor.fetchone()[0]
                        cursor.execute("UPDATE parkeer_algemeen SET maxaantalplaatsen = ? where parkingid = ?", maxplaatsen,i)
                        cursor.commit()

                        cursor.execute("UPDATE parkeer_algemeen set aantalvrijeplaatsen = ? where parkingid=?", vrijeplaatsen, int(i))
                        cursor.commit()

                    video_capture.release()
                    cv2.destroyAllWindows()

                    if foto != '':
                        try:
                            print("Nu gaan we alles verwijderen")
                            top = str(local_path / i)
                            for root, dirs, files in os.walk(top, topdown=False):
                                for name in files:
                                    filename = os.path.join(root, name)
                                    os.chmod(filename, stat.S_IWUSR)
                                    os.remove(filename)
                                for name in dirs:
                                    os.rmdir(os.path.join(root, name))
                            os.rmdir(top)
                            os.mkdir(str(local_path / i))
                            print("alles is verwijderd")
                            break
                        except (PermissionError,OSError):
                            print("Kon foto " + i + " niet verwijderen.")

for i in os.listdir(local_path):
    seconden = nu.timestamp() - os.path.getmtime(local_path / i)
    minuten = seconden / 60
    #wanneer tijd groter dan 60 minuten -> camera "kapot": 0 = kapot
    if minuten > 60:
        cursor.execute("UPDATE parkeer_algemeenstippen set status = 0 where parkingid = ?", i)
    else:
        cursor.execute("UPDATE parkeer_algemeenstippen set status = 1 where parkingid = ?", i)
    cursor.commit()