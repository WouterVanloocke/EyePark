from django.http import HttpResponse
from parkeer.models import Analysis,Algemeen
from django.shortcuts import render, redirect
from django.views.generic.base import TemplateView
from django.db.models import Avg
from datetime import datetime
from parkeer.forms import HomeForm
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views import generic
from parkeer.forms import HomeForm
import os
import pyodbc
import ast
import time

#Database connection
server = 'arxusparking.database.windows.net'
database = 'arxusparkingDB'
username = 'stage'
password = 'Wachtwoord123'
driver= '{SQL Server}'
cnxn = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password)
cursor = cnxn.cursor()


class HomePageView(TemplateView):
    template_name = 'index.html'

    def get(self, request):
        analyse = Analysis.objects.all()
        algemeen = Algemeen.objects.all()

        form = HomeForm()

        plaats = []
        bezetteplaats = []
        vrijeplaatsen = []
        plaatsbezet = []
        aantalvrijeplaatsen = []
        aantalbezetteplaatsen = []
        statussen = []

        def ValuesQuerySetToDict(vqs):
            return [item for item in vqs]

        gemiddeldePlaatsen = ValuesQuerySetToDict(Analysis.objects.values_list('tijd') \
         .annotate(gemiddeldAantalPlaatsen=Avg('aantalvrijeplaatsen'))\
         .order_by('tijd'))

        datum = datetime.now()
        local_path = ('d:/home/foto/')
        #local_path = ('C:/Users/Robin.De.Bie/Documents/Scripts/foto/')
        retry_flag = True
        retry_count = 0

        #for i in os.listdir(local_path):
        #    seconden = datum.timestamp() - os.path.getmtime(local_path + i)
        #    minuten = seconden / 60
        #    #wanneer tijd groter dan 60 minuten -> camera "kapot": 0 = kapot
        #    if minuten > 60:
        #        #while retry_flag and retry_count < 5:
        #        #  try:
        #        cursor.execute("UPDATE parkeer_algemeen set status = 0 where parkingid = ?", i)
        #            #retry_flag = False
        #          #except:
        #          #  retry_count = retry_count + 1
        #          #  time.sleep(1)
        #        kapot = 0
        #    else:
        #        #while retry_flag and retry_count < 5:
        #        #  try:
        #        cursor.execute("UPDATE parkeer_algemeen set status = 1 where parkingid = ?", i)
        #          #  retry_flag = False
        #          #except:
        #          #  retry_count = retry_count + 1
        #          #  time.sleep(1)
        #        kapot = 1

        #    status.append(kapot)
        for i in algemeen:
            status = i.status
            statussen.append(status)

        for i in algemeen:
            if(i.status == 1):
                vrijeplaats = i.aantalvrijeplaatsen
                vrijeplaatsen.append(vrijeplaats)
                bezet = i.maxaantalplaatsen - i.aantalvrijeplaatsen
                plaatsbezet.append(bezet)
            else:
                vrijeplaatsen.append(0)
                plaatsbezet.append(0)

        aantalvrijplaatsen = sum(vrijeplaatsen)
        bezetteplaatsen = sum(plaatsbezet)

        #cursor.commit();

        return render(request, 'index.html', {'gemiddeldePlaatsen': gemiddeldePlaatsen, 'plaatsbezet' : plaatsbezet, 'vrijeplaatsen' : vrijeplaatsen, 'bezetteplaatsen': bezetteplaatsen, 'aantalvrijplaatsen' : aantalvrijplaatsen, 'statussen': statussen, 'form': form})

#### ophalen Pushberichten scriptfiles
class manifest(TemplateView):
    template_name = 'manifest.json'

class OneSignalSDKWorker(TemplateView):
    template_name = 'OneSignalSDKWorker.js'

class OneSignalSDKUpdaterWorker(TemplateView):
    template_name = 'OneSignalSDKUpdaterWorker.js'

#html pagina's definiëren
class camera1(TemplateView):
    template_name = 'camera1.html'
    def get(self, request):
        return render(request, 'camera1.html')

class camera2(TemplateView):
    template_name = 'camera2.html'
    def get(self, request):
        return render(request, 'camera2.html')

class kaart(TemplateView):
    template_name = 'kaart.html'
    def get(self, request):
        analyse = Analysis.objects.all()
        algemeen = Algemeen.objects.all()

        form = HomeForm()

        plaats = []
        bezetteplaats = []
        vrijeplaatsen = []
        plaatsbezet = []
        aantalvrijeplaatsen = []
        aantalbezetteplaatsen = []
        statussen = []

        def ValuesQuerySetToDict(vqs):
            return [item for item in vqs]

        gemiddeldePlaatsen = ValuesQuerySetToDict(Analysis.objects.values_list('tijd') \
         .annotate(gemiddeldAantalPlaatsen=Avg('aantalvrijeplaatsen'))\
         .order_by('tijd'))

        datum = datetime.now()
        local_path = ('d:/home/foto/')
        retry_flag = True
        retry_count = 0
        for i in algemeen:
            status = i.status
            statussen.append(status)

        for i in algemeen:
            if(i.status == 1):
                vrijeplaats = i.aantalvrijeplaatsen
                vrijeplaatsen.append(vrijeplaats)
                bezet = i.maxaantalplaatsen - i.aantalvrijeplaatsen
                plaatsbezet.append(bezet)
            else:
                vrijeplaatsen.append(0)
                plaatsbezet.append(0)

        aantalvrijplaatsen = sum(vrijeplaatsen)
        bezetteplaatsen = sum(plaatsbezet)

        return render(request, 'kaart.html', {'gemiddeldePlaatsen': gemiddeldePlaatsen, 'plaatsbezet' : plaatsbezet, 'vrijeplaatsen' : vrijeplaatsen, 'bezetteplaatsen': bezetteplaatsen, 'aantalvrijplaatsen' : aantalvrijplaatsen, 'statussen': statussen, 'form': form})

class allecameras(TemplateView):
    template_name = 'allecameras.html'
    def get(self, request):
        algemeen = Algemeen.objects.all()
        def ValuesQuerySetToDict(vqs):
            return [item for item in vqs]
        camerabeschrijvingen = ValuesQuerySetToDict(Algemeen.objects.values_list('parkingid', 'beschrijving').order_by('parkingid'))
        return render(request, 'allecameras.html', {"camerabeschrijvingen": camerabeschrijvingen})

class grid(LoginRequiredMixin, generic.CreateView):
    template_name = 'grid.html'
    def get(self,request):
        form = HomeForm()
        return render(request, self.template_name, {'form': form})

    def post(self, request):
        form = HomeForm(request.POST)
        if form.is_valid():
           beschrijving = form.cleaned_data['beschrijving']
           rectx = form.cleaned_data['rectx']
           recty = form.cleaned_data['recty']
           rectwidth = form.cleaned_data['rectwidth']
           rectheight = form.cleaned_data['rectheight']
           camx = form.cleaned_data['camx']
           camy = form.cleaned_data['camy']
           camrichting = form.cleaned_data['camrichting']
           cursor.execute("INSERT INTO parkeer_rasterwaarde(rectx , recty, rectwidth, rectheight, camx, camy, camrichting, beschrijving) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rectx, recty, rectwidth, rectheight, camx, camy, camrichting, beschrijving)
           cursor.commit();
           return redirect('grid')
        args = {'form': form, 'rectx': rectx, 'recty': recty, 'rectwidth': rectwidth, 'rectheight': rectheight, 'camx' : camx, 'camy': camy, 'camrichting' : camrichting, 'beschrijving' : beschrijving}
        return render(request, self.template_name, args)