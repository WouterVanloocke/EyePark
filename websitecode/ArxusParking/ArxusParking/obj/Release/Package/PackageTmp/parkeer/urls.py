from django.urls import path
from django.conf.urls import url
from django.contrib import admin
from django.contrib.auth.views import LoginView, LogoutView
from django.contrib.auth.decorators import login_required
from django.views.generic import TemplateView
from app import forms, views
from datetime import datetime
from . import views

urlpatterns = [
    path('', views.HomePageView.as_view(), name='index'),
    url(r'^camera1$', views.camera1.as_view(), name='camera1'),
    url(r'^camera2$', views.camera2.as_view(), name='camera2'),
    url(r'^allecameras$', views.allecameras.as_view(), name='allecameras'),
    url(r'^admin/', admin.site.urls),
    path('grid/', views.grid.as_view(), name='grid'),
]