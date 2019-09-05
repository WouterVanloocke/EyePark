from django import forms
from parkeer.models import Rasterwaarde

class HomeForm(forms.ModelForm):
   keuzes = [('',''), ('camT', 'Naar boven'), ('camR', 'Naar rechts'), ('camB', 'Naar beneden'), ('camL', 'Naar onder')]

   beschrijving = forms.CharField(widget=forms.TextInput({
                                  'class': 'form-control',
                                  'placeholder': 'beschrijving'}))
   rectx = forms.IntegerField(widget=forms.TextInput({
                                  'class': 'form-control',
                                  'placeholder': 'x'}))
   recty = forms.IntegerField(widget=forms.TextInput({
                                  'class': 'form-control',
                                  'placeholder': 'y'}))
   rectwidth = forms.IntegerField(widget=forms.TextInput({
                                  'class': 'form-control',
                                  'placeholder': 'width'}))
   rectheight = forms.IntegerField(widget=forms.TextInput({
                                  'class': 'form-control',
                                  'placeholder': 'height'}))
   camx = forms.IntegerField(widget=forms.TextInput({
                                  'class': 'form-control',
                                  'placeholder': 'camera x'}))
   camy = forms.IntegerField(widget=forms.TextInput({
                                  'class': 'form-control',
                                  'placeholder': 'camera y'}))
   camrichting = forms.ChoiceField(choices = keuzes, widget=forms.Select({'class': 'form-control'}))

   class Meta:
       model = Rasterwaarde
       fields = ('beschrijving','rectx', 'recty', 'rectwidth', 'rectheight', 'camx', 'camy', 'camrichting',)