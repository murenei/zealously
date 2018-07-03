from django.urls import path

'''
The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
    Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
'''

from . import views
from .views import ListUsers, ListQuestions

urlpatterns = [
    path('', views.index, name='index'),
    path('gen_equation', views.gen_equation, name='gen_equation'),

    # REST API Calls
    path('api/users/', ListUsers.as_view(), name='user-list'),
    path('api/questions/', ListQuestions.as_view(), name='question-list'),
]
