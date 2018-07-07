"""zealot URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')

"""

from django.urls import path
from . import views
from django.views.generic import TemplateView

# set applicatoin namespace for url namespacing
app_name = 'textapp'

urlpatterns = [
    path('', views.index, name='index'),
    path('analyse_sentiment', views.analyse_sentiment, name='analyse-sentiment'),
    path('spam', views.detect_spam, name='detect-spam'),
    path('similarity', views.CalcSimilarity.as_view(), name='calc-similarity'),

    path('detect-faces', views.DetectFaces.as_view(), name='detect-faces'),
    # path('similarity', views.calc_similarity, name='calc-similarity'),
    # path('', TemplateView.as_view(template_name='home.html'), name='home'),
    # the template engine doesn't search the 'mysite' (zealot) folder for templates. It does search all of the app folders though.
    # For now, home.html is in the mathapp template folder. Will eventually create a separate app for the home pages.
    # path('about', AboutView.as_view(), name='about-view')


]
