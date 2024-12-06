from django.contrib.auth.models import AbstractBaseUser, BaseUserManager
from django.db import models

# Custom user manager to handle user creation and superuser creation
class CustomUserManager(BaseUserManager):
    def create_user(self, username, password=None, **extra_fields):
        if not username:
            raise ValueError('El usuario debe tener un nombre de usuario')
        user = self.model(username=username, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, username, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        return self.create_user(username, password, **extra_fields)

# Custom user model
class User(AbstractBaseUser):
    username = models.CharField(unique=True, max_length=150)
    email = models.EmailField(unique=True, null=True, blank=True)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)  # Permite que el usuario sea admin
    date_joined = models.DateTimeField(auto_now_add=True)

    objects = CustomUserManager()

    USERNAME_FIELD = 'username'
    REQUIRED_FIELDS = ['email']  # Campos obligatorios para crear un superusuario

    def __str__(self):
        return self.username

# Perfil del usuario con la imagen facial
class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    face_image = models.ImageField(upload_to='user_faces/', null=True, blank=True)

    def __str__(self):
        return f"Profile of {self.user.username}"

# Modelo para almacenar los resultados de ML
class MLResult(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # Relación con el usuario
    model_name = models.CharField(max_length=100)  # Nombre del modelo de ML utilizado
    accuracy = models.FloatField(null=True, blank=True)  # Métrica de precisión
    predictions = models.JSONField()  # Predicciones en formato JSON
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"MLResult for {self.user.username} using {self.model_name}"
