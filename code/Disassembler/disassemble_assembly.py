import os
import subprocess
import sys
import tempfile

def run_ghidra_decompilation(file_path, temp_project_path, output_dir):
    # Configura i percorsi necessari
    ghidra_path = "/Users/matteotiozzo/Desktop/ghidra_11.1.2_PUBLIC"
    analyze_headless = os.path.join(ghidra_path, "support/analyzeHeadless")
    script_path = os.path.join(ghidra_path, "Ghidra/Features/Decompiler/ghidra_scripts")
    
    # Determina i nomi base del file
    file_base_name = os.path.basename(file_path)
    log_file_path = os.path.join(output_dir, "{}_ghidra.log".format(file_base_name))
    
    # Output per l'assembly di questo eseguibile
    assembly_output_path = os.path.join(output_dir, "{}_assembly.txt".format(file_base_name))

    # Costruisci il comando
    command = [
        analyze_headless,
        temp_project_path, "tempProject", "-import", file_path,
        "-postScript", "DecompileAllFunctions.py", assembly_output_path,
        "-scriptPath", script_path,
        "-processor", "x86:LE:64:default",
        "-log", log_file_path
    ]

    # Esegui il comando
    try:
        subprocess.check_call(command)
        print("Decompilazione completata con successo per il file: {}".format(file_path))
    except subprocess.CalledProcessError as e:
        print("Errore durante la decompilazione:", str(e))
        return

    # Verifica se il file assembly è stato creato correttamente
    if os.path.exists(assembly_output_path):
        print("File assembly salvato in: {}".format(assembly_output_path))
    else:
        print("File assembly non trovato: {}".format(assembly_output_path))

def process_all_files(directory, output_dir):
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):
            # Crea un progetto Ghidra temporaneo per ciascun file
            with tempfile.TemporaryDirectory() as temp_project_path:
                run_ghidra_decompilation(file_path, temp_project_path, output_dir)

if __name__ == "__main__":
    malware_directory = "/Users/matteotiozzo/Desktop/Thesis/code/Malware/malware_dataset_virushare"
    output_dir = "/Users/matteotiozzo/Desktop/Thesis/code/Labeling/Logs/"
    process_all_files(malware_directory, output_dir)
