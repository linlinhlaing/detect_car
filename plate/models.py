from django.db import models

# Create your models here.
class uploadimage(models.Model): 
    upload_Img = models.ImageField(upload_to='uploads/') 