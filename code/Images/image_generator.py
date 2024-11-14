import numpy as np
from PIL import Image
import os

# Percorsi delle cartelle
input_dir = '16x16_matrix_hex/Virus'
output_dir = '16x16_grayscale_images_hex/Virus'

# Crea la cartella di output se non esiste
os.makedirs(output_dir, exist_ok=True)

# Itera su tutti i file .txt nella cartella di input
for filename in os.listdir(input_dir):
    if filename.endswith('.txt'):
        file_path = os.path.join(input_dir, filename)
        
        # Carica il file txt e crea la matrice
        matrice = np.loadtxt(file_path, dtype=int)

        # Applica la Min-Max normalization per portare i valori tra 0 e 255
        min_val = matrice.min()
        max_val = matrice.max()
        normalized_matrice = ((matrice - min_val) / (max_val - min_val) * 255).astype(np.uint8)

        # Crea l'immagine in scala di grigi
        image = Image.fromarray(normalized_matrice, mode='L')
        
        # Salva l'immagine nella cartella di output con lo stesso nome del file txt
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.png")
        image.save(output_path)
        
        print(f"Immagine salvata in '{output_path}'")
