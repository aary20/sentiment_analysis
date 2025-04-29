from celery import shared_task
from django.contrib.auth.models import User
from .models import UserAnswer
from .utils.email_utils import send_performance_report
from datetime import timedelta
from django.utils import timezone

@shared_task
def send_weekly_reports():
    users = User.objects.all()
    for user in users:
        one_week_ago = timezone.now() - timedelta(days=7)
        user_answers = UserAnswer.objects.filter(user=user, timestamp__gte=one_week_ago)

        if user_answers.exists():
            send_performance_report(user, user_answers)
