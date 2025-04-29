from django.core.mail import send_mail
from django.template.loader import render_to_string

def send_performance_report(user, user_answers):
    subject = f"Your Interview Preparation Weekly Report, {user.first_name}"
    message = render_to_string('emails/weekly_report.html', {
        'user': user,
        'user_answers': user_answers
    })
    send_mail(
        subject,
        '',  # plain text (optional)
        'your-email@gmail.com',
        [user.email],
        html_message=message
    )
