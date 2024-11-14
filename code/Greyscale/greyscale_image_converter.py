from PIL import Image
import numpy as np
import os
from tqdm import tqdm  # Importa tqdm per la barra di progresso

def text_to_grayscale_image(text_file_path, image_width):
    # Leggi il contenuto del file di testo
    with open(text_file_path, 'r') as text_file:
        text_data = text_file.read()

    # Converti i caratteri in valori interi (ad esempio ASCII)
    binary_data = [ord(char) for char in text_data]

    # Calcola il numero di pixel possibili
    total_pixels = len(binary_data)

    # Determina l'altezza dell'immagine basata sulla larghezza specificata
    image_height = total_pixels // image_width
    if total_pixels % image_width != 0:
        image_height += 1  # Aggiungi una riga se necessario per ospitare tutti i pixel

    # Crea un array numpy dai dati e ridimensiona per rappresentare l'immagine
    image_data = np.array(binary_data, dtype=np.uint8)
    image_data = np.pad(image_data, (0, image_height * image_width - len(image_data)), mode='constant')
    image_data = image_data.reshape((image_height, image_width))

    # Crea l'immagine utilizzando PIL, specificando 'L' per scala di grigi
    image = Image.fromarray(image_data, 'L')

    # Costruisci il nome del file di output e il percorso nella cartella 'Greyscale'
    output_file_name = os.path.splitext(os.path.basename(text_file_path))[0]
    output_file_name = output_file_name.replace("_assembly", "") + '.png'
    output_file_path = os.path.join("images", output_file_name)  # Percorso aggiornato

    # Salva l'immagine
    image.save(output_file_path)

# Assicurati che la cartella 'Greyscale' esista nel percorso corretto
greyscale_directory = os.path.join("images")
if not os.path.exists(greyscale_directory):
    os.makedirs(greyscale_directory)

# Processa tutti i file nella cartella 'Assembly'
directory = "../Labeling/assembly"
files_to_process = [f for f in os.listdir(directory) if f.endswith(".txt")]  # Filtra i file .txt

# Aggiungi la barra di progresso durante il processing dei file
for filename in tqdm(files_to_process, desc="Converting text to images", unit="file"):
    file_path = os.path.join(directory, filename)
    text_to_grayscale_image(file_path, 256)  # Imposta la larghezza a 256 pixel
