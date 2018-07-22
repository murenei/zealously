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
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path
from django.http import HttpResponsePermanentRedirect
from mathapp.views import AboutView
from django.views.generic import TemplateView


import requests


def airtable_confirm(request, record_id):
    url = "https://api.airtable.com/v0/appHgC3xduUgUe4EI/Agenda%20Template%20copy/"
    record_id = record_id
    url += record_id
    headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer keyJmtv8qAZgap79u'}
    payload = {'fields': {'Confirmed': 'YES!'}}
    r = requests.patch(url, headers=headers, json=payload)
    print(r.url)
    print(r.text)

    return HttpResponsePermanentRedirect("https://www.google.com")


urlpatterns = [
    path('admin/', admin.site.urls),
    path('math/', include('mathapp.urls'), name='mathapp'),
    path('text/', include('textapp.urls'), name='textapp'),
    path('', TemplateView.as_view(template_name='mathapp/home.html'), name='home'),
    # the template engine doesn't search the 'mysite' (zealot) folder for templates. It does search all of the app folders though.
    # For now, home.html is in the mathapp template folder. Will eventually create a separate app for the home pages.
    path('about', AboutView.as_view(), name='about-view'),



    # and for airtablw
    path('airtable/confirm/<str:record_id>', airtable_confirm, name='airtable_confirm')



]
