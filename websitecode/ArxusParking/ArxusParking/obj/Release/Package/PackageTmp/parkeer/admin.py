from django.contrib import admin
from parkeer.models import Algemeen,Analysis,Exacteplaats,Plaatstemp,Rasterwaarde,Analysis
# Register your models here.

# Define the admin class
class AlgemeenAdmin(admin.ModelAdmin):
    list_display = ('id', 'parkingid', 'aantalvrijeplaatsen', 'maxaantalplaatsen', 'status','beschrijving','maxbereikt')

class ExacteplaatsAdmin(admin.ModelAdmin):
    list_display = ('id', 'xcoördinaat', 'ycoördinaat', 'parkingid')

class PlaatstempAdmin(admin.ModelAdmin):
    list_display = ('id', 'xcoördinaat', 'ycoördinaat', 'parkingid')

class AnalysisAdmin(admin.ModelAdmin):
    list_display = ('id', 'parkingid', 'aantalvrijeplaatsen', 'tijd')

class RasterwaardeAdmin(admin.ModelAdmin):
    list_display = ('id', 'rectx', 'recty', 'rectwidth', 'rectheight', 'camx', 'camy', 'camrichting')

# Register the admin class with the associated model
admin.site.register(Algemeen, AlgemeenAdmin)
admin.site.register(Exacteplaats, ExacteplaatsAdmin)
admin.site.register(Plaatstemp, PlaatstempAdmin)
admin.site.register(Rasterwaarde, RasterwaardeAdmin)
admin.site.register(Analysis, AnalysisAdmin)

