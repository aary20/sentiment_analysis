# sentiment_app/urls.py
from django.urls import path
from . import views

app_name = 'sentiment_app'

urlpatterns = [
    path('', views.index, name='index'),
    path('visualize/', views.visualize, name='visualize'),
    path('predict/', views.predict, name='predict'),
]
