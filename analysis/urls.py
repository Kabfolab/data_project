from django.urls import path
from . import views

urlpatterns = [
    
    path('data_analysis/', views.data_analysis, name='data_analysis'),
]
