from django.urls import path
from .views import authenticate_user, register_user, LoginAPIView, CSVUploadView, ejecutar_ml

urlpatterns = [
    path('api/face-login/', authenticate_user, name='face-login'),  # Vista basada en función
    path('api/register/', register_user, name='register'),  # Vista basada en función
    path('api/login/', LoginAPIView.as_view(), name='login'),  # Vista basada en clase (correcta)
    path('api/upload-csv/', CSVUploadView.as_view(), name='csv-upload'),  # Vista basada en clase
    path('api/ml/', ejecutar_ml, name='ml'),  # Vista basada en función
]
