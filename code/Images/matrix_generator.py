import numpy as np
import os
import pandas as pd

# Carica il file CSV
file_path = 'output_mnemonics_hex/Virus_mnemonics_occurrences.csv'
df = pd.read_csv(file_path, header=None)

# Crea la cartella "matrici" se non esiste
output_dir = "16x16_matrix_hex/Virus"
os.makedirs(output_dir, exist_ok=True)

# Itera su ciascuna riga del DataFrame, a partire dalla seconda riga
for i in range(1, len(df)):
    # Estrai il nome del malware dalla prima colonna
    malware_name = df.iloc[i, 0]

    # Estrai i valori della riga a partire dalla seconda colonna
    values = df.iloc[i, 1:].values

    # Crea una matrice 16x16 con tutti 0
    matrice = np.zeros((16, 16), dtype=int)

    # Popola la matrice con i valori estratti
    for idx, value in enumerate(values):
        row = idx // 16  # Determina la riga attuale
        col = idx % 16   # Determina la colonna attuale
        if row < 16:     # Assicura di non superare i limiti della matrice 16x16
            matrice[row, col] = value

    # Salva la matrice in un file .txt con il nome del malware
    output_path = os.path.join(output_dir, f"{malware_name}.txt")
    np.savetxt(output_path, matrice, fmt="%d")

    print(f"Matrice per '{malware_name}' salvata in '{output_path}'")
