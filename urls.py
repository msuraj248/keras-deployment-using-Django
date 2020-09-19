from django.contrib import admin
from django.urls import path
from kerasapp import views

urlpatterns = [
    path('', views.index, name = 'kerasapp'),
    path('prediction', views.prediction, name = 'prediction')
]