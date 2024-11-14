# ######################################################################################
# # Per creare il grafico della varianza dei primi 60 elementi principali
# import pandas as pd
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt

# data = pd.read_csv('tfidf_table_main_assembly.csv')

# data_numeric = data.drop(columns=['Path', 'Malware'])

# pca = PCA(n_components=60)  
# components = pca.fit_transform(data_numeric)

# explained_variance = pca.explained_variance_ratio_

# cumulative_variance = explained_variance.cumsum()

# plt.figure(figsize=(10, 6))
# plt.bar(range(1, 61), explained_variance, alpha=0.6, label='Varianza spiegata per componente', color='b')
# plt.plot(range(1, 61), cumulative_variance, marker='o', linestyle='-', color='r', label='Varianza spiegata cumulativa')
# plt.xlabel('Componenti Principali')
# plt.ylabel('Varianza Spiegata')
# plt.title('Varianza Spiegata e Cumulativa dei Primi 60 Componenti Principali')
# plt.axhline(y=0.95, color='g', linestyle='--', label='95% Varianza Spiegata')
# plt.legend()
# plt.grid(True)
# plt.savefig('primi_60_componenti.png', bbox_inches='tight')

######################################################################################



#######################################################################################
## Per plottare PC1 vs PC2, PC1 vs PC3, PC2 vs PC3, PC1 vs PC4 e PC2 vs PC4 ecc..
# import pandas as pd
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

# # Carica i dati da un file CSV
# data = pd.read_csv('tfidf_table_main_assembly.csv')

# # Rimuove le colonne non numeriche
# data_numeric = data.drop(columns=['Malware', 'Path'])  # Assume 'Path' and 'Malware' are le colonne non-numeriche

# # Applica PCA direttamente sui dati
# pca = PCA()
# components = pca.fit_transform(data_numeric)

# # Crea un DataFrame con i componenti principali
# components_df = pd.DataFrame(components, columns=[f'PC{i+1}' for i in range(data_numeric.shape[1])])

# # Calcola e salva la varianza spiegata in percentuale
# explained_variance = pca.explained_variance_ratio_ * 100  # Varianza in percentuale
# with open('principal_components1.txt', 'w') as f:
#     f.write("Componenti Principali Ordinati per Importanza (Varianza Spiegata in %):\n")
#     for i, variance in enumerate(explained_variance, start=1):
#         f.write(f"PC{i}: Varianza Spiegata = {variance:.2f}%\n")

# # Salva le caratteristiche e i coefficienti per ciascun componente principale
# with open('principal_component_coefficients.txt', 'w') as f:
#     for i, component in enumerate(pca.components_):
#         f.write(f"\nPrincipal Component {i+1}:\n")
#         for feature, coefficient in zip(data_numeric.columns, component):
#             f.write(f"{feature}: {coefficient:.4f}\n")

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

# # Estrai la famiglia di malware dal campo 'Path'
# def extract_family(path):
#     for family in malware_colors:
#         if family in path:
#             return family
#     return "Other"

# # Applica la funzione per creare una colonna di etichette
# data['Family'] = data['Path'].apply(extract_family)

# # Assegna colori a seconda della famiglia
# colors = data['Family'].map(malware_colors).fillna('grey')

# # Genera grafici dei componenti principali
# plt.figure(figsize=(12, 10))

# # Creazione di un plot per ogni set di componenti
# for i, (pc1, pc2) in enumerate([(1, 2), (1, 3), (2, 3), (1, 4)], start=1):
#     ax = plt.subplot(2, 2, i)
#     for family, color in malware_colors.items():
#         # Filtra per famiglia
#         idx = data['Family'] == family
#         ax.scatter(components_df.loc[idx, f'PC{pc1}'], components_df.loc[idx, f'PC{pc2}'], c=color, label=family, alpha=0.5)
#     plt.title(f'PC{pc1} vs PC{pc2}')
#     plt.xlabel(f'PC{pc1}')
#     plt.ylabel(f'PC{pc2}')

# plt.legend(title="Malware Family", bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.savefig('principal_components1.png', bbox_inches='tight')
#######################################################################################


import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Carica i dati da un file CSV
data = pd.read_csv('tfidf_table_main_hex.csv')

# Rimuove le colonne non numeriche
data_numeric = data.drop(columns=['Malware', 'Path'])  # Assume 'Path' e 'Malware' sono colonne non numeriche

# Applica PCA direttamente sui dati
pca = PCA()
components = pca.fit_transform(data_numeric)

# Crea un DataFrame con i componenti principali
components_df = pd.DataFrame(components, columns=[f'PC{i+1}' for i in range(data_numeric.shape[1])])

# Calcola e salva la varianza spiegata in percentuale
explained_variance = pca.explained_variance_ratio_ * 100  # Varianza in percentuale
with open('principal_components1.txt', 'w') as f:
    f.write("Componenti Principali Ordinati per Importanza (Varianza Spiegata in %):\n")
    for i, variance in enumerate(explained_variance, start=1):
        f.write(f"PC{i}: Varianza Spiegata = {variance:.2f}%\n")

# Salva le caratteristiche e i coefficienti per ciascun componente principale
with open('principal_component_coefficients_hex.txt', 'w') as f:
    for i, component in enumerate(pca.components_):
        f.write(f"\nPrincipal Component {i+1}:\n")
        for feature, coefficient in zip(data_numeric.columns, component):
            f.write(f"{feature}: {coefficient:.4f}\n")

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

# Estrai la famiglia di malware dal campo 'Path'
def extract_family(path):
    for family in malware_colors:
        if family in path:
            return family
    return "Other"

# Applica la funzione per creare una colonna di etichette
data['Family'] = data['Path'].apply(extract_family)

# Assegna colori a seconda della famiglia
colors = data['Family'].map(malware_colors).fillna('grey')

# Grafici individuali dei componenti principali
plt.figure(figsize=(12, 10))

# Creazione di un plot per ogni set di componenti
for i, (pc1, pc2) in enumerate([(1, 2), (1, 3), (2, 3), (1, 4)], start=1):
    ax = plt.subplot(2, 2, i)
    for family, color in malware_colors.items():
        idx = data['Family'] == family
        ax.scatter(components_df.loc[idx, f'PC{pc1}'], components_df.loc[idx, f'PC{pc2}'], c=color, label=family, alpha=0.5)
    plt.title(f'PC{pc1} vs PC{pc2}')
    plt.xlabel(f'PC{pc1}')
    plt.ylabel(f'PC{pc2}')

plt.legend(title="Malware Family", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('principal_components_individual.png', bbox_inches='tight')

# Grafico combinato dei primi quattro componenti principali
plt.figure(figsize=(10, 8))
for family, color in malware_colors.items():
    idx = data['Family'] == family
    plt.scatter(components_df.loc[idx, 'PC1'], components_df.loc[idx, 'PC2'], c=color, label=family, alpha=0.5)
    plt.scatter(components_df.loc[idx, 'PC3'], components_df.loc[idx, 'PC4'], c=color, alpha=0.5)

plt.title('Primi 4 Componenti Principali')
plt.xlabel('Componenti Principali')
plt.ylabel('Valori dei Componenti')
plt.legend(title="Malware Family", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('principal_components_combined.png', bbox_inches='tight')
