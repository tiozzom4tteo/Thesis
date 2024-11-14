import os
import shutil

# Lista dei percorsi da ignorare
ignored_paths = [
    "mnemonic_assembly_superfamily/Adware",
    "mnemonic_assembly_superfamily/Backdoor",
    "mnemonic_assembly_superfamily/Downloader",
    "mnemonic_assembly_superfamily/Ransomware",
    "mnemonic_assembly_superfamily/Spyware",
    "mnemonic_assembly_superfamily/Trojan",
    "mnemonic_assembly_superfamily/Virus"
]

# Funzione per navigare nella directory e cercare e verificare il file "subfolder_data.txt"
def delete_empty_mnemonics_folders(root_folder):
    for dirpath, dirnames, filenames in os.walk(root_folder, topdown=False):
        # Controlla se il percorso corrente deve essere ignorato
        if any(dirpath.endswith(ignored) for ignored in ignored_paths):
            print(f"Skipping protected folder: {dirpath}")
            continue  # Salta al prossimo ciclo se il percorso è protetto
        
        # Controlla se "subfolder_data.txt" esiste nella cartella
        if "subfolder_mnemonics.txt" in filenames:
            file_path = os.path.join(dirpath, "subfolder_mnemonics.txt")
            # Controlla se il file è vuoto
            if os.path.getsize(file_path) == 0:
                # Cancella la cartella con tutti i file al suo interno
                print(f"Deleting folder: {dirpath}")
                shutil.rmtree(dirpath)
            else:
                # Se il file non è vuoto, non fare nulla
                print(f"Skipping folder, file not empty: {dirpath}")
        else:
            # Se il file non esiste, non fare nulla
            print(f"Skipping folder, file not found: {dirpath}")

# Esegui la funzione sulla directory desiderata
root_folder = "mnemonic_assembly_superfamily/"
delete_empty_mnemonics_folders(root_folder)
