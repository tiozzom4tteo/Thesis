import os
import subprocess

cartella_reports = 'report'
cartella_avclass = 'avclass'

if not os.path.exists(cartella_avclass):
    os.makedirs(cartella_avclass)

for nome_file in os.listdir(cartella_reports):
    if nome_file.endswith('.json'):
        percorso_file = os.path.join(cartella_reports, nome_file)
        
        nome_file_output = os.path.splitext(nome_file)[0] + '.txt'
        percorso_file_output = os.path.join(cartella_avclass, nome_file_output)
        
        comando = ['avclass', '-f', percorso_file, '-o', percorso_file_output]
        
        with open(percorso_file_output, 'w') as file_output:
            subprocess.run(comando, stdout=file_output, stderr=subprocess.STDOUT)

print(f"Processo completato. I risultati sono stati generati nella cartella {cartella_avclass}.")