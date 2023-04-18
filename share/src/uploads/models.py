from django.db import models
from .utils import get_filtered_image
from PIL import Image
import numpy as np
from io import BytesIO
from django.core.files.base import ContentFile

# Create your models here.

ACTION_CHOICES = (
    ('NO_FILTER', 'No filter'),
    ('COLOURIZED', 'HSV Format'),
    ('GRAYSCALE', 'Grayscale'),
    ('BLURRED', 'Blurred'),
    ('BINARY', 'Binary'),
    ('INVERT', 'Invert'),
    ('EDGE_DETECTION', 'Edge Detection'),
    ('LINE_DETECTION', 'Line Detection'),
    ('FACE_DETECTION', 'Face Detection'),
    ('SHAPE_DETECTION', 'Shape Detection'),
    ('IMAGE_DENOISING', 'Image Denoising'),
    ('IMAGE_MORPHOLOGY', 'Image Morphology'),
    ('WHITE_BALANCING','White Balancing'),
    ('TEXT_DETECTION', 'Text Detection'),
    ('TEXT_REMOVAL', 'Text Removal')
   
    

    

    
)

class Upload(models.Model):
    image = models.ImageField(upload_to = 'images')
    action = models.CharField(max_length=100, choices= ACTION_CHOICES)
    updated = models.DateTimeField(auto_now = True)
    created = models.DateTimeField(auto_now_add= True)


    def __str__(self):
        return str(self.id)

    def save(self, *args, **kwargs):
        #open image
        pil_img = Image.open(self.image)

        #convert the image and do some processing
        cv_img = np.array(pil_img)
        img = get_filtered_image(cv_img, self.action)

        #convert image to pil image
        im_pil = Image.fromarray(img)

        #save
        buffer = BytesIO()
        im_pil.save(buffer, format = 'png')
        image_png = buffer.getvalue()

        self.image.save(str(self.image), ContentFile(image_png), save = False)

        super().save(*args, **kwargs)
         