from django.contrib.messages import constants as messages


INSTALLED_APPS = [
    # Default apps
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django_celery_beat',
    'core',
]
MESSAGE_TAGS = {
    messages.ERROR: 'danger',
}

# Redirect after login/logout
LOGIN_REDIRECT_URL = 'dashboard'
LOGOUT_REDIRECT_URL = 'login'

# Add media settings
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_USE_TLS = True
EMAIL_HOST_USER = 'arrmkn09@gmail.com'
EMAIL_HOST_PASSWORD = '@aarya209'  # NOT your normal password!
DEFAULT_FROM_EMAIL = EMAIL_HOST_USER

CELERY_BROKER_URL = 'redis://localhost:6379'
