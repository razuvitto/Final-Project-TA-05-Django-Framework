# Coded with <3 Razuvitto
# location : FPGrowth/urls.py
# 2018-2019

from django.urls import path
from FPGrowth import views

app_name = 'FPGrowth'
urlpatterns = [
    path('', views.index),
    path('frequent', views.frequent),
    path('closed', views.closed),
    path('rules', views.rules),
    path('mainpage', views.mainpage),
    path('optimizerules', views.optimizerules),
    path('compareresult', views.compareresult),
    path('binning-data', views.documentationbinning),
    path('data-cleaning', views.datacleaning),
    path('data-selection', views.dataselection),
    path('data-transformation', views.datatransformation)
]

# End of file