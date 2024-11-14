import pandas as pd

# Carica il file e leggi i dati
input_file = "../Occurrences/mnemonics_summary.txt"
# input_file = "../Occurrences/hex_summary.txt"
output_file = "filtered_mnemonics_summary.txt"
# output_file = "filtered_hex_summary.txt"

# Leggi il file saltando la prima riga con i nomi delle colonne e altre righe superflue
df = pd.read_csv(input_file, sep="\s{2,}", engine='python', skiprows=3, names=["Mnemonic-Pair", "Total Occurrence/Frequency", "Total Files"])

# Filtra le righe in base al valore della seconda colonna
df_filtered = df[df["Total Occurrence/Frequency"] >= 200]

# Salva il file modificato
df_filtered.to_csv(output_file, sep="\t", index=False)

print(f"File filtrato salvato come: {output_file}")
