import os
import pandas as pd
import numpy as np


main_directory = "mnemonic_assembly_superfamily"  
# main_directory = "mnemonic_hex_without_empty_files"  

mnemonics_summary_path = "filtered_mnemonics_summary.txt"
# mnemonics_summary_path = "filtered_hex_summary.txt"

file_txt = "subfolder_mnemonics.txt"
# file_txt = "subfolder_data.txt"

table_name = "tfidf_table_main_assembly.csv"
# table_name = "tfidf_table_main_hex.csv"

# Leggi il file mnemonics_summary.txt per ottenere le coppie di mnemonici
mnemonics_summary = []
with open(mnemonics_summary_path, "r") as file:
    for line in file.readlines()[1:]:  # Salta la prima riga di intestazione
        parts = line.split()
        if len(parts) >= 2:
            mnemonic_pair = f"{parts[0]} {parts[1]}"
            mnemonics_summary.append(mnemonic_pair)

# Crea la tabella principale con le coppie di mnemonici come colonne
df_tfidf_main = pd.DataFrame(columns=["Path", "Malware"] + mnemonics_summary)

# Naviga all'interno delle sottocartelle e trova i file "subfolder_mnemonics"
all_pairs_count = []
malware_paths = []

for root, dirs, files in os.walk(main_directory):
    for file in files:
        if file == file_txt:
            file_path = os.path.join(root, file)
            malware_name = os.path.basename(root)
            
            # Leggi il file subfolder_mnemonics e raccogli i dati per TF-IDF
            pairs_count = {}
            with open(file_path, "r") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 3:
                        pair = f"{parts[0]} {parts[1]}"
                        count = int(parts[2])
                        pairs_count[pair] = count

            all_pairs_count.append(pairs_count)
            malware_paths.append((file_path, malware_name))

# Calcola l'IDF per ogni coppia di mnemonici
num_documents = len(all_pairs_count)
idf_values = {}

for mnemonic_pair in mnemonics_summary:
    # Conta in quanti documenti appare la coppia di mnemonici
    doc_count = sum(1 for pairs_count in all_pairs_count if mnemonic_pair in pairs_count)
    # Calcola l'IDF, usando logaritmo naturale e aggiungendo 1 al denominatore per evitare divisioni per zero
    idf_values[mnemonic_pair] = np.log((num_documents + 1) / (doc_count + 1)) + 1

# Popola la tabella principale con i valori TF-IDF
for idx, (pairs_count, (file_path, malware_name)) in enumerate(zip(all_pairs_count, malware_paths)):
    tfidf_values = [file_path, malware_name]  # La prima colonna contiene il percorso, la seconda il nome del malware
    for mnemonic_pair in mnemonics_summary:
        tf = pairs_count.get(mnemonic_pair, 0)
        idf = idf_values.get(mnemonic_pair, 0)
        tfidf = tf * idf
        tfidf_values.append(tfidf)
    
    # Aggiungi la riga al DataFrame principale
    df_tfidf_main.loc[len(df_tfidf_main)] = tfidf_values

# Salva la tabella in un file CSV
df_tfidf_main.to_csv(table_name, index=False)
