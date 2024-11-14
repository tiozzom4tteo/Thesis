# Percorso del file di input
input_file_path = 'principal_component_coefficients_hex.txt'
# input_file_path = 'principal_component_coefficients.txt'
# Percorso del file di output
# output_file_path = 'sorted_principal_component_coefficients.txt'
output_file_path = 'sorted_principal_component_coefficients_hex.txt'

# Inizializza un dizionario per memorizzare i dati
components_data = {}

# Legge il file e organizza i dati
with open(input_file_path, 'r') as file:
    current_component = None
    for line in file:
        # Controlla se la linea identifica una componente principale
        if line.startswith("Principal Component"):
            current_component = line.strip()
            components_data[current_component] = []
            print(f"Trovata nuova componente: {current_component}")  # Debug
        elif current_component and line.strip():  # Ignora le righe vuote
            try:
                # Divide e memorizza l'istruzione e il valore
                instruction, value = line.split(':')
                instruction = instruction.strip()
                value = float(value.strip())
                components_data[current_component].append((instruction, value))
                print(f"Aggiunta coppia: ({instruction}, {value}) alla {current_component}")  # Debug
            except ValueError as e:
                print(f"Errore di parsing nella linea: {line}. Errore: {e}")  # Debug

# Ordina le coppie per valore in ordine decrescente all'interno di ciascun componente
sorted_components = {
    comp: sorted(pairs, key=lambda x: x[1], reverse=True)
    for comp, pairs in components_data.items()
}

# Salva i risultati ordinati nel file di output
with open(output_file_path, 'w') as file:
    for component, sorted_pairs in sorted_components.items():
        file.write(f"{component}:\n")
        for instruction, value in sorted_pairs:
            file.write(f"  {instruction}: {value:.4f}\n")
        file.write("\n")
    print(f"Risultati ordinati salvati in '{output_file_path}'")  # Debug

print("Processo completato.")
