from django.contrib import admin
from django.urls import path, include
from django.http import HttpResponse
from django.conf import settings
from django.conf.urls.static import static
from ml import views


def home(request):
    return HttpResponse("Welcome to the homepage!")

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home, name='home'),  # Página de inicio básica
    path('', include('ml.urls')),  # Todas las rutas de `ml` estarán bajo el prefijo `api/`
    path('csrf/', views.csrf, name='csrf'),

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
