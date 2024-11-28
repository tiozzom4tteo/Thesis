import matplotlib.pyplot as plt
import numpy as np
import re

# Funzione di smoothing
def smooth(data, window_size=10):
    """Applica una media mobile per ridurre il rumore."""
    if len(data) == 0:
        raise ValueError("La lista fornita per il smoothing è vuota.")
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Funzione per sostituire valori mancanti
def fill_missing_values(data):
    """Sostituisce i valori None con l'ultimo valore valido."""
    filled_data = []
    last_valid = None
    for value in data:
        if value is not None:
            filled_data.append(value)
            last_valid = value
        else:
            filled_data.append(last_valid)  # Usa l'ultimo valore valido
    return filled_data

# Percorso del file
file_path = 'metrics_1000_32batch/evaluation_metrics.txt'

# Variabili per memorizzare i dati
epochs = []
blackbox_accuracy = []
blackbox_precision = []
blackbox_f1 = []
noise_level = []

# Parsing del file
with open(file_path, 'r') as file:
    current_epoch = None
    current_noise = None
    current_blackbox_accuracy = None
    current_blackbox_precision = None
    current_blackbox_f1 = None

    for line in file:
        line = line.strip()
        
        # Cerca l'epoca
        epoch_match = re.search(r'Epoch (\d+)', line)
        if epoch_match:
            # Salva i dati raccolti precedentemente
            if current_epoch is not None:
                epochs.append(current_epoch)
                noise_level.append(current_noise)
                blackbox_accuracy.append(current_blackbox_accuracy)
                blackbox_precision.append(current_blackbox_precision)
                blackbox_f1.append(current_blackbox_f1)

            # Inizia un nuovo record
            current_epoch = int(epoch_match.group(1))
            current_noise = None
            current_blackbox_accuracy = None
            current_blackbox_precision = None
            current_blackbox_f1 = None

        # Cerca i valori di interesse
        noise_match = re.search(r'Noise Level:\s*([\d.]+)', line)
        if noise_match:
            current_noise = float(noise_match.group(1))

        bb_acc_match = re.search(r'Blackbox Accuracy:\s*([\d.]+)', line)
        if bb_acc_match:
            current_blackbox_accuracy = float(bb_acc_match.group(1))

        bb_prec_match = re.search(r'Precision:\s*([\d.]+)', line)
        if bb_prec_match and 'Blackbox' in line:  # Assicura che sia per il Blackbox
            current_blackbox_precision = float(bb_prec_match.group(1))

        bb_f1_match = re.search(r'F1:\s*([\d.]+)', line)
        if bb_f1_match and 'Blackbox' in line:  # Assicura che sia per il Blackbox
            current_blackbox_f1 = float(bb_f1_match.group(1))

    # Salva l'ultimo record
    if current_epoch is not None:
        epochs.append(current_epoch)
        noise_level.append(current_noise)
        blackbox_accuracy.append(current_blackbox_accuracy)
        blackbox_precision.append(current_blackbox_precision)
        blackbox_f1.append(current_blackbox_f1)

# Riempie i valori mancanti
blackbox_accuracy = fill_missing_values(blackbox_accuracy)
blackbox_precision = fill_missing_values(blackbox_precision)
blackbox_f1 = fill_missing_values(blackbox_f1)
noise_level = fill_missing_values(noise_level)

# Sincronizzazione finale di tutte le liste
all_data = list(zip(epochs, blackbox_accuracy, blackbox_precision, blackbox_f1, noise_level))
all_data = [record for record in all_data if None not in record]

# Decomposizione delle liste sincronizzate
synced_epochs, synced_blackbox_accuracy, synced_blackbox_precision, synced_blackbox_f1, synced_noise_level = zip(*all_data)

# Applica il smoothing
window_size = 20
smoothed_epochs = synced_epochs[:len(synced_epochs) - window_size + 1]
smoothed_accuracy = smooth(synced_blackbox_accuracy, window_size)
smoothed_precision = smooth(synced_blackbox_precision, window_size)
smoothed_f1 = smooth(synced_blackbox_f1, window_size)
smoothed_noise = smooth(synced_noise_level, window_size)

# Debug: verifica che le liste siano sincronizzate
print(f"Lunghezza smoothed_epochs: {len(smoothed_epochs)}")
print(f"Lunghezza smoothed_accuracy: {len(smoothed_accuracy)}")
print(f"Lunghezza smoothed_precision: {len(smoothed_precision)}")
print(f"Lunghezza smoothed_f1: {len(smoothed_f1)}")
print(f"Lunghezza smoothed_noise: {len(smoothed_noise)}")

# Creazione del grafico
plt.figure(figsize=(12, 8))

plt.plot(smoothed_epochs, smoothed_accuracy, label='Blackbox Accuracy', linestyle='-', marker=None)
plt.plot(smoothed_epochs, smoothed_f1, label='Blackbox F1 Score', linestyle='-.', marker=None)
plt.plot(smoothed_epochs, smoothed_noise, label='Noise Level', linestyle=':', marker=None)

# Aggiunta di etichette e legenda
plt.xlabel('Epochs')
plt.ylabel('Metric Value')
plt.title('Smoothed Blackbox Metrics and Noise Level Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('metrics_1000_32batch/evaluation_metrics_smoothed.png')
