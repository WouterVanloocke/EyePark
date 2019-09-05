from django.db import models

class Analysis(models.Model):
   parkingid = models.IntegerField(blank=True, null=True)
   aantalvrijeplaatsen = models.IntegerField(blank=True, null=True)
   tijd = models.IntegerField(blank=True, null=True)

class Algemeen(models.Model):
   parkingid = models.IntegerField(blank=True, null=True)
   aantalvrijeplaatsen = models.IntegerField(blank=True, null=True)
   maxaantalplaatsen = models.IntegerField(blank=True, null=True)
   status = models.IntegerField(blank=True, null=True)
   beschrijving = models.CharField(blank=True, null=True, max_length = 200)
   maxbereikt = models.CharField(blank=True, null=True, max_length = 10)

class Rasterwaarde(models.Model):
   rectx = models.IntegerField(blank=True, null=True)
   recty = models.IntegerField(blank=True, null=True)
   rectwidth = models.IntegerField(blank=True, null=True)
   rectheight = models.IntegerField(blank=True, null=True)
   camx = models.IntegerField(blank=True, null=True)
   camy = models.IntegerField(blank=True, null=True)
   camrichting = models.CharField(max_length = 10, blank=True, null=True)

class Exacteplaats(models.Model):
    xcoördinaat = models.IntegerField(blank=True, null=True)
    ycoördinaat = models.IntegerField(blank=True, null=True)
    parkingid = models.IntegerField(blank=True, null=True)

class Plaatstemp(models.Model):
    xcoördinaat = models.IntegerField(blank=True, null=True)
    ycoördinaat = models.IntegerField(blank=True, null=True)
    parkingid = models.IntegerField(blank=True, null=True)