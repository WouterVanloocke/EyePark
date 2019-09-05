"""
Definition of urls for ArxusParking.
"""

from datetime import datetime
from django.urls import path,include
from django.contrib import admin
from django.contrib.auth.views import LoginView, LogoutView
from app import forms, views
from django.conf.urls import include, url
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic.base import TemplateView
from django.views.generic.base import RedirectView
from django.contrib.auth import views as auth_views
import parkeer.views

favicon_view = RedirectView.as_view(url='/static/Arxus.ico', permanent=True)

urlpatterns = [
    url(r'^',include('parkeer.urls')),
    #url(r'^$', parkeer.views.index, name='index'),
    #url(r'^home$', parkeer.views.index, name='home'),
    path('', views.home, name='home'),
    #path('contact/', views.contact, name='contact'),
    #path('about/', views.about, name='about'),
    path('accounts/', include('django.contrib.auth.urls')), # new
    path('login/',
         LoginView.as_view
         (
             template_name='accounts/login.html',
             authentication_form=forms.BootstrapAuthenticationForm,
             extra_context=
             {
                 'title': 'Log in',
                 'year' : datetime.now().year,
             }
         ),
         name='login'),
    path('logout/', LogoutView.as_view(next_page='/'), name='logout'),
    path('admin/', admin.site.urls),
    path('favicon.ico', favicon_view),
    path('manifest.json', TemplateView.as_view(template_name='manifest.json', content_type='application/json')),
    path('OneSignalSDKWorker.js', TemplateView.as_view(template_name='OneSignalSDKWorker.js', content_type='application/x-javascript')),
    path('OneSignalSDKUpdaterWorker.js', TemplateView.as_view(template_name='OneSignalUpdaterSDKWorker.js', content_type='application/x-javascript')),

] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
