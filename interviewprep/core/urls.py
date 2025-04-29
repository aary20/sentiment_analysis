from django.urls import path
from . import views
from core import views

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', views.login_user, name='login_user'),
    path('register/', views.register_user, name='register_user'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('upload_resume/', views.upload_resume, name='upload_resume'),
    path('mock-interview/', views.mock_interview, name='mock_interview'),
    path('reports/', views.reports, name='reports'),
    path('generate-questions/', views.generate_questions, name='generate_questions'),
    path('mock-interview/', views.mock_interview, name='mock_interview'),
    path('thank-you/', views.thank_you, name='thank_you'),
    path('analyze_answer/', views.analyze_answer, name='analyze_answer'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('report/<int:id>/', views.view_report, name='view_report'),

]