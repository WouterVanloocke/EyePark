import pyodbc
import requests

#Databankconstanten
server = 'arxusparking.database.windows.net'
database = 'arxusparkingDB'
username = 'stage'
password = 'Wachtwoord123'
driver= '{SQL Server}'
cnxn = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password)
cursor = cnxn.cursor()

tekst = ""
cursor.execute("SELECT parkingid, aantalvrijeplaatsen FROM parkeer_algemeen") 
vrijeplaatsen = cursor.fetchall()

for i in vrijeplaatsen:
    if i[1] == 1:
        lijn = "Er is nog " +  str(i[1]) + " vrije plaats op parking " + str(i[0]) + '.\n'
        tekst += lijn
    elif i[1] > 1:
        lijn = "Er zijn nog " + str(i[1]) + " vrije plaatsen op parking " + str(i[0]) + '.\n'
        tekst += lijn

data = {
    "app_id": "bc7c9c93-994e-4894-9cc0-8d8fee67b4c0",
    "included_segments": ["All"],
    "contents": {"en": tekst}
}

requests.post(
    "https://onesignal.com/api/v1/notifications",
    headers={"Authorization": "Basic ZTc5OWQ2YWQtYzAyNi00N2VhLWEzOGYtNjNiZWZmNmRhMGVj"},
    json=data
)