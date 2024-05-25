from PIL import Image
import os
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image


def resize_images(folder_path, new_size=(224, 224)):
    # Obtener la lista de archivos en la carpeta
    files = os.listdir(folder_path)
    output_dir = 'C:/Users/grupo_gepar/Documents/lucho/Osteo/Processing/Valid/Ostheo'
    # Inicializar contador para el número en el nombre de la imagen
    num = 1
    
    # Iterar sobre cada archivo en la carpeta
    for file_name in files:
        # Verificar si el archivo es una imagen
        if file_name.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            # Construir la ruta completa del archivo
            file_path = os.path.join(folder_path, file_name)
            
            # Abrir la imagen
            img = Image.open(file_path)
            
            # Cambiar el tamaño de la imagen
            resized_img = img.resize(new_size)
            
            # Construir el nuevo nombre de la imagen redimensionada
            new_name = f"resized_ostheo_val_{num}"
            
            # Guardar la imagen redimensionada con el nuevo nombre
            resized_img.save(os.path.join(output_dir, new_name + ".png"))
            
            # Incrementar el contador
            num += 1

# Ruta de la carpeta que contiene las imágenes
folder_path = "C:/Users/grupo_gepar/Documents/lucho/Osteo/Dataset/Valid/Valid/Osteoarthritis"

# Cambiar el tamaño de las imágenes en la carpeta y cambiarles el nombre
resize_images(folder_path)

