from django import forms 
from .models import *
  
class UploadForm(forms.ModelForm): 
  
    class Meta: 
        model = uploadimage 
        fields = ['upload_Img'] 