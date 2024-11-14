#######################################################################################
# With standardization
import os
import shutil
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

table_name = "tfidf_table_main_assembly.csv"
# table_name = "tfidf_table_main_hex.csv"

table_name_output = "filtered_pca_tfidf_table_standardizzata_assembly.csv"
# table_name_output = "filtered_pca_tfidf_table_standardizzata_hex.csv"

main_directory = "mnemonic_assembly_without_empty_files"  
# main_directory = "mnemonic_hex_without_empty_files"  

file_txt = "subfolder_mnemonics.txt"
# file_txt = "subfolder_data.txt"

image_name = "filtered_pca_plot_standardizzata_assembly.png"
# image_name = "filtered_pca_plot_standardizzata_hex.png"


# Carica la tabella TF-IDF generata in precedenza
df_tfidf_main = pd.read_csv(table_name)

# Seleziona solo le colonne delle caratteristiche (escludendo 'Path' e 'Malware')
features = df_tfidf_main.drop(columns=["Path", "Malware"]).astype(float)

# Standardizza le caratteristiche
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Applica il PCA per ridurre le dimensioni a 2 componenti
pca = PCA(n_components=2)
principal_components = pca.fit_transform(features_scaled)

most_influential_features = [features.columns[i] for i in pca.components_[0].argsort()[::-1][:2]]

# Crea un DataFrame per le componenti principali
df_pca = pd.DataFrame(data=principal_components, columns=["Principal Component 1", "Principal Component 2"])
df_pca["Malware"] = df_tfidf_main["Malware"].values
df_pca["Path"] = df_tfidf_main["Path"].values

# Calcola i limiti di percentile per le componenti principali
pc1_lower, pc1_upper = np.percentile(df_pca["Principal Component 1"], [20, 80])
pc2_lower, pc2_upper = np.percentile(df_pca["Principal Component 2"], [20, 80])

# Filtra il DataFrame in base ai percentili
filtered_df = df_pca[
    (df_pca["Principal Component 1"] >= pc1_lower) & (df_pca["Principal Component 1"] <= pc1_upper) &
    (df_pca["Principal Component 2"] >= pc2_lower) & (df_pca["Principal Component 2"] <= pc2_upper)
]

# Salva la tabella filtrata in un file CSV
filtered_df.to_csv(table_name_output, index=False)

# Ottieni l'insieme dei percorsi dei file filtrati
filtered_files = set(filtered_df["Path"])

# Rimuovi le cartelle che non contengono "subfolder_mnemonics.txt" in filtered_files
excluded_dirs = {
    "mnemonic_assembly_without_empty_files/Backdoor",
    "mnemonic_assembly_without_empty_files/Trojan",
    "mnemonic_assembly_without_empty_files/Downloader",
    "mnemonic_assembly_without_empty_files/Ransomware",
    "mnemonic_assembly_without_empty_files/Spyware",
    "mnemonic_assembly_without_empty_files/Virus",
    "mnemonic_assembly_without_empty_files/Adware"
}

# for root, dirs, files in os.walk(main_directory):
#     for dir_name in dirs:
#         dir_path = os.path.join(root, dir_name)
#         # Normalizza il percorso per confrontarlo correttamente
#         normalized_dir_path = os.path.normpath(dir_path)
#         if normalized_dir_path not in excluded_dirs:
#             subfolder_file_path = os.path.normpath(os.path.join(dir_path, file_txt))
#             if subfolder_file_path not in filtered_files:
#                 print(f"Rimuovi cartella {dir_path}")
#                 shutil.rmtree(dir_path)

# Mappa dei colori per le famiglie di malware
malware_colors = {
    "Adware": "red",
    "Backdoor": "blue",
    "Trojan": "green",
    "Downloader": "purple",
    "Ransomware": "orange",
    "Spyware": "brown",
    "Virus": "pink"
}

# Visualizza e salva le componenti principali dopo il filtraggio
plt.figure(figsize=(10, 6))
for index, row in filtered_df.iterrows():
    malware = row["Malware"]
    for family in malware_colors:
        if family in row["Path"]:
            color = malware_colors[family]
            plt.scatter(row["Principal Component 1"], row["Principal Component 2"], color=color)
            break

# Aggiungi una legenda con un solo punto per famiglia di malware
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=family) 
           for family, color in malware_colors.items()]
plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))

# Aggiungi spiegazione della varianza
explained_variance = pca.explained_variance_ratio_
plt.xlabel(f"{most_influential_features[0]} ({explained_variance[0]*100:.2f}% explained variance)")
plt.ylabel(f"{most_influential_features[1]} ({explained_variance[1]*100:.2f}% explained variance)")
plt.title("Mnemonic pair PCA")
plt.tight_layout()
plt.savefig(image_name, bbox_inches='tight')

## Salva il modello PCA e lo scaler per uso futuro
# import joblib
# joblib.dump(pca, 'pca_model.pkl')
# joblib.dump(scaler, 'scaler.pkl')
######################################################################################











#######################################################################################
# # Without standardization
# import os
# import shutil
# import pandas as pd
# import numpy as np
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

# table_name = "tfidf_table_main_assembly.csv"
# # table_name = "tfidf_table_main_hex.csv"

# table_name_output = "test.csv"
# # table_name_output = "filtered_pca_tfidf_table_hex.csv"

# main_directory = "mnemonic_assembly_without_empty_files"  
# # main_directory = "mnemonic_hex_without_empty_files"  

# file_txt = "subfolder_mnemonics.txt"
# # file_txt = "subfolder_data.txt"

# image_name = "test.png"
# # image_name = "filtered_pca_plot_hex.png"

# # Carica la tabella TF-IDF generata in precedenza
# df_tfidf_main = pd.read_csv(table_name)

# # Seleziona solo le colonne delle caratteristiche (escludendo 'Path' e 'Malware')
# features = df_tfidf_main.drop(columns=["Path", "Malware"]).astype(float)

# # Applica il PCA per ridurre le dimensioni a 2 componenti
# pca = PCA(n_components=2)
# principal_components = pca.fit_transform(features)

# most_influential_features = [features.columns[i] for i in pca.components_[0].argsort()[::-1][:2]]

# # Crea un DataFrame per le componenti principali
# df_pca = pd.DataFrame(data=principal_components, columns=["Principal Component 1", "Principal Component 2"])
# df_pca["Malware"] = df_tfidf_main["Malware"].values
# df_pca["Path"] = df_tfidf_main["Path"].values

# # Salva la tabella filtrata in un file CSV
# df_pca.to_csv(table_name_output, index=False)

# # Ottieni l'insieme dei percorsi dei file filtrati
# filtered_files = set(df_pca["Path"])

# # Rimuovi le cartelle che non contengono "subfolder_mnemonics.txt" in filtered_files
# excluded_dirs = {
#     "mnemonic_assembly_without_empty_files/Backdoor",
#     "mnemonic_assembly_without_empty_files/Trojan",
#     "mnemonic_assembly_without_empty_files/Downloader",
#     "mnemonic_assembly_without_empty_files/Ransomware",
#     "mnemonic_assembly_without_empty_files/Spyware",
#     "mnemonic_assembly_without_empty_files/Virus",
#     "mnemonic_assembly_without_empty_files/Adware"
# }

# # for root, dirs, files in os.walk(main_directory):
# #     for dir_name in dirs:
# #         dir_path = os.path.join(root, dir_name)
# #         # Normalizza il percorso per confrontarlo correttamente
# #         normalized_dir_path = os.path.normpath(dir_path)
# #         if normalized_dir_path not in excluded_dirs:
# #             subfolder_file_path = os.path.normpath(os.path.join(dir_path, file_txt))
# #             if subfolder_file_path not in filtered_files:
# #                 print(f"Rimuovi cartella {dir_path}")
# #                 shutil.rmtree(dir_path)

# # Mappa dei colori per le famiglie di malware
# malware_colors = {
#     "Adware": "red",
#     "Backdoor": "blue",
#     "Trojan": "green",
#     "Downloader": "purple",
#     "Ransomware": "orange",
#     "Spyware": "brown",
#     "Virus": "pink"
# }

# # Visualizza e salva le componenti principali
# plt.figure(figsize=(10, 6))
# for index, row in df_pca.iterrows():
#     malware = row["Malware"]
#     for family in malware_colors:
#         if family in row["Path"]:
#             color = malware_colors[family]
#             plt.scatter(row["Principal Component 1"], row["Principal Component 2"], color=color)
#             break

# # Aggiungi una legenda con un solo punto per famiglia di malware
# handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=family) 
#            for family, color in malware_colors.items()]
# plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))

# # Aggiungi spiegazione della varianza
# explained_variance = pca.explained_variance_ratio_
# plt.xlabel(f"{most_influential_features[0]} ({explained_variance[0]*100:.2f}% explained variance)")
# plt.ylabel(f"{most_influential_features[1]} ({explained_variance[1]*100:.2f}% explained variance)")
# plt.title("Mnemonic pair PCA")
# plt.tight_layout()
# plt.savefig(image_name, bbox_inches='tight')

#######################################################################################




#######################################################################################
#senza un cazzo

# import os
# import shutil
# import pandas as pd
# import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt

# table_name = "tfidf_table_main_assembly.csv"
# # table_name = "tfidf_table_main_hex.csv"

# table_name_output = "filtered_pca_tfidf_table_standardizzata_assembly1.csv"
# # table_name_output = "filtered_pca_tfidf_table_standardizzata_hex.csv"

# main_directory = "mnemonic_assembly_without_empty_files"  
# # main_directory = "mnemonic_hex_without_empty_files"  

# file_txt = "subfolder_mnemonics.txt"
# # file_txt = "subfolder_data.txt"

# image_name = "filtered_pca_plot_standardizzata_assembly1.png"
# # image_name = "filtered_pca_plot_standardizzata_hex.png"

# # Carica la tabella TF-IDF generata in precedenza
# df_tfidf_main = pd.read_csv(table_name)

# # Seleziona solo le colonne delle caratteristiche (escludendo 'Path' e 'Malware')
# features = df_tfidf_main.drop(columns=["Path", "Malware"]).astype(float)

# # Standardizza le caratteristiche
# scaler = StandardScaler()
# features_scaled = scaler.fit_transform(features)

# # Applica il PCA per ridurre le dimensioni
# pca_full = PCA()
# pca_full.fit(features_scaled)

# # Calcola la varianza spiegata per tutte le componenti principali
# explained_variance_full = pca_full.explained_variance_ratio_

# # Crea un Scree Plot
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(explained_variance_full) + 1), explained_variance_full, marker='o', linestyle='--')
# plt.xlabel('Numero di Componenti Principali')
# plt.ylabel('Varianza Spiegata')
# plt.title('Scree Plot')
# plt.grid(True)
# plt.savefig("scree_plot.png", bbox_inches='tight')

# # Seleziona il numero di componenti principali da usare in base a una soglia di varianza
# total_variance_threshold = 0.80
# cumulative_variance = np.cumsum(explained_variance_full)
# num_components = np.argmax(cumulative_variance >= total_variance_threshold) + 1
# print(f'Numero di componenti principali da prendere: {num_components}')

# # Applica il PCA con il numero selezionato di componenti
# pca = PCA(n_components=num_components)
# principal_components = pca.fit_transform(features_scaled)

# most_influential_features = [features.columns[i] for i in pca.components_[0].argsort()[::-1][:2]]

# # Crea un DataFrame per le componenti principali
# df_pca = pd.DataFrame(data=principal_components, columns=[f"Principal Component {i+1}" for i in range(num_components)])
# df_pca["Malware"] = df_tfidf_main["Malware"].values
# df_pca["Path"] = df_tfidf_main["Path"].values

# # Salva la tabella filtrata in un file CSV
# df_pca.to_csv(table_name_output, index=False)

# # Ottieni l'insieme dei percorsi dei file filtrati
# filtered_files = set(df_pca["Path"])

# # Mappa dei colori per le famiglie di malware
# malware_colors = {
#     "Adware": "red",
#     "Backdoor": "blue",
#     "Trojan": "green",
#     "Downloader": "purple",
#     "Ransomware": "orange",
#     "Spyware": "brown",
#     "Virus": "pink"
# }

# # Visualizza e salva le componenti principali
# plt.figure(figsize=(10, 6))
# for index, row in df_pca.iterrows():
#     malware = row["Malware"]
#     for family in malware_colors:
#         if family in row["Path"]:
#             color = malware_colors[family]
#             plt.scatter(row["Principal Component 1"], row["Principal Component 2"], color=color)
#             break

# # Aggiungi una legenda con un solo punto per famiglia di malware
# handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=family) 
#            for family, color in malware_colors.items()]
# plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))

# # Aggiungi spiegazione della varianza
# explained_variance = pca.explained_variance_ratio_
# plt.xlabel(f"{most_influential_features[0]} ({explained_variance[0]*100:.2f}% explained variance)")
# plt.ylabel(f"{most_influential_features[1]} ({explained_variance[1]*100:.2f}% explained variance)")
# plt.title("Mnemonic pair PCA")
# plt.tight_layout()
# plt.savefig(image_name, bbox_inches='tight')

#######################################################################################