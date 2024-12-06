from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import FileUploadParser
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework import status
from django.contrib.auth import get_user_model, authenticate
from django.http import JsonResponse
from .models import UserProfile
import cv2
import numpy as np
from deepface import DeepFace
from rest_framework.decorators import api_view
from rest_framework.permissions import AllowAny

class CSVUploadView(APIView):
    parser_class = (FileUploadParser,)

    def post(self, request, *args, **kwargs):
        try:
            # Verifica si el archivo está en la solicitud
            if 'file' not in request.FILES:
                return Response({'error': 'No file provided'}, status=status.HTTP_400_BAD_REQUEST)

            file = request.FILES['file']
            df = pd.read_csv(file)

            # Verifica que las columnas requeridas existan
            if 'some_column' not in df.columns or 'target_column' not in df.columns:
                return Response({'error': 'Columns "some_column" and "target_column" are required in the dataset.'},
                                status=status.HTTP_400_BAD_REQUEST)

            # Divide los datos y entrena el modelo
            X = df[['some_column']]
            y = df['target_column']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)

            prediction = model.predict(X_test)

            return Response({'prediction': prediction.tolist()}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Vista para la autenticación facial
@api_view(['POST'])
def authenticate_user(request):
    if request.method == "POST":
        # Obtener la imagen enviada por el usuario (la imagen de su rostro)
        uploaded_face = request.FILES.get('face_image')
        if not uploaded_face:
            return JsonResponse({"message": "No se ha proporcionado una imagen de rostro."}, status=400)

        # Leer la imagen en memoria
        img_array = np.frombuffer(uploaded_face.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Obtener el nombre de usuario de los datos recibidos
        username = request.POST.get('username')
        if not username:
            return JsonResponse({"message": "El nombre de usuario es obligatorio."}, status=400)

        # Buscar al usuario y su perfil
        try:
            user = get_user_model().objects.get(username=username)
            profile = UserProfile.objects.get(user=user)
        except (get_user_model().DoesNotExist, UserProfile.DoesNotExist):
            return JsonResponse({"message": "Usuario o perfil no encontrados."}, status=404)

        # Obtener la imagen facial guardada en el perfil del usuario
        stored_face_image_path = profile.face_image.path

        # Usar DeepFace para comparar las dos imágenes
        try:
            result = DeepFace.verify(img, stored_face_image_path)
            if result['verified']:
                return JsonResponse({"message": "Autenticación exitosa."})
            else:
                return JsonResponse({"message": "La imagen no coincide con el rostro registrado."}, status=401)
        except Exception as e:
            return JsonResponse({"message": f"Error en el reconocimiento facial: {str(e)}"}, status=500)

    # Si el método no es POST, se retorna un error
    return JsonResponse({"message": "Método no permitido."}, status=405)

# Vista para la autenticación por nombre de usuario y contraseña
@api_view(['POST'])
def authenticate_with_password(request):
    if request.method == "POST":
        # Obtener el nombre de usuario y la contraseña de los datos recibidos
        username = request.data.get('username')
        password = request.data.get('password')

        # Intentar autenticar al usuario
        user = authenticate(username=username, password=password)
        if user is not None:
            refresh = RefreshToken.for_user(user)
            return Response({
                'access': str(refresh.access_token),
                'refresh': str(refresh),
            })
        else:
            return JsonResponse({'message': 'Credenciales inválidas'}, status=400)

    return JsonResponse({"message": "Método no permitido."}, status=405)

# Vista para el login mediante JWT
class LoginAPIView(APIView):
    permission_classes = [AllowAny]  # Permite acceso sin necesidad de autenticación

    def post(self, request, *args, **kwargs):
        username = request.data.get("username")
        password = request.data.get("password")

        user = authenticate(username=username, password=password)

        if user is not None:
            refresh = RefreshToken.for_user(user)
            return Response({
                'access': str(refresh.access_token),
                'refresh': str(refresh),
            })
        return Response({"detail": "Invalid credentials."}, status=status.HTTP_401_UNAUTHORIZED)

# Vista para registrar un nuevo usuario
@api_view(['POST'])
def register_user(request):
    if request.method == "POST":
        username = request.data.get('username')
        password = request.data.get('password')
        face_image = request.FILES.get('face_image')  # Captura la imagen facial desde la solicitud

        if not username or not password or not face_image:
            return JsonResponse({"message": "Todos los campos son requeridos."}, status=400)

        # Crear el usuario
        try:
            user = get_user_model().objects.create_user(username=username, password=password)
            user_profile = UserProfile.objects.create(user=user, face_image=face_image)  # Guardar la imagen facial

            return JsonResponse({"message": "Usuario registrado con éxito."})
        except Exception as e:
            return JsonResponse({"message": f"Error al registrar el usuario: {str(e)}"}, status=500)
    
    return JsonResponse({"message": "Método no permitido"}, status=405)

import subprocess
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
import tempfile
import shutil

@csrf_exempt
def ejecutar_ml(request):
    if request.method == 'POST':
        # Obtener el archivo CSV y las variables del formulario
        file = request.FILES.get('file')
        variables = request.POST.get('variables')  # Asumimos que es un JSON con las variables

        # Guardar el archivo en un directorio temporal
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, 'input.csv')
        
        # Guardar el archivo CSV en disco
        with open(file_path, 'wb') as f:
            for chunk in file.chunks():
                f.write(chunk)
        
        # Pasar la ruta del archivo al script de Python para la ejecución
        python_script_path = os.path.join(temp_dir, 'ml_script.py')
        
        # Aquí podrías escribir tu código de ML en Python en un archivo temporal
        with open(python_script_path, 'w') as f:
            f.write("""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Cargar los datos con codificación adecuada
df = pd.read_csv('/content/Fuentes_Financiamiento_ONPvf.csv', encoding='ISO-8859-1')

# Limpiar y preparar los datos
df['PERIODO'] = df['PERIODO'].astype(str)
df['PERIODO'] = df['PERIODO'].str.replace(r'\.0$', '', regex=True)
df['AÑO'] = df['PERIODO'].str[:4].astype(int)
df['MES'] = df['PERIODO'].str[4:].astype(int)

# Seleccionar las variables predictoras y la variable dependiente
X = df[['AÑO', 'MES', 'RECURSOS ORDINARIOS', 'CONTRIBUCIONES A FONDOS']]
y = df['TOTAL']

# Dividir el dataset en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)

# Entrenar el modelo
rf_regressor.fit(X_train, y_train)

# Predicciones en el conjunto de prueba
y_pred = rf_regressor.predict(X_test)

# Evaluar el modelo utilizando diferentes métricas
mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
rmse = np.sqrt(mse)  # Root Mean Squared Error
r2 = r2_score(y_test, y_pred)  # R² Score

# Mostrar las métricas
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R2 Score: {r2}')

# Obtener la importancia de las características
importancia = rf_regressor.feature_importances_
print(f'Importancia de las características: {importancia}')

# Generar los datos para los períodos futuros (hasta 2032)
futuro_periodos = []
for año in range(2023, 2033):  # Años hasta 2032
    for mes in range(1, 13):  # Meses del 1 al 12
        futuro_periodos.append([año, mes, 0, 0])  # Asumiendo que los recursos ordinarios y contribuciones a fondos son 0

futuro_df = pd.DataFrame(futuro_periodos, columns=['AÑO', 'MES', 'RECURSOS ORDINARIOS', 'CONTRIBUCIONES A FONDOS'])

# Predecir el total para los períodos futuros
futuro_predicciones = rf_regressor.predict(futuro_df)

# Agregar las predicciones al DataFrame futuro
futuro_df['TOTAL_PREDICCION'] = futuro_predicciones

# Mostrar las predicciones hasta el 2032
print(futuro_df)

# Graficar los resultados
plt.figure(figsize=(10, 6))
plt.plot(X_test['AÑO'] + X_test['MES'] / 100, y_test, color='blue', marker='o', linestyle='-', label='Valor Real')  # 'PERIODO' como número
plt.plot(X_test['AÑO'] + X_test['MES'] / 100, y_pred, color='red', marker='x', linestyle='--', label='Predicción')  # 'PERIODO' como número
plt.xlabel('PERIODO (Año + Mes)')
plt.ylabel('TOTAL')
plt.title('Valor Real vs Predicción')
plt.legend()
plt.show()

""" % (file_path, os.path.join(temp_dir, 'predicciones.txt'), os.path.join(temp_dir, 'grafico.png')))

        # Ejecutar el script de Python
        try:
            subprocess.run(['python3', python_script_path], check=True)
        except subprocess.CalledProcessError as e:
            return JsonResponse({'error': str(e)}, status=500)

        # Leer el archivo de resultados y devolver la respuesta
        with open(os.path.join(temp_dir, 'predicciones.txt'), 'r') as f:
            predicciones = f.read()

        with open(os.path.join(temp_dir, 'grafico.png'), 'rb') as f:
            grafico = f.read()

        # Limpiar los archivos temporales
        shutil.rmtree(temp_dir)

        # Devolver los resultados
        return JsonResponse({
            'predicciones': predicciones,
            'grafico': grafico.decode('utf-8')  # Opción para devolver como base64 si se necesita
        })