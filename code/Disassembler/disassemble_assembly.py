import os
import subprocess
import tempfile

def run_ghidra_decompilation(file_path, temp_project_path, output_dir):
    # Configura i percorsi necessari
    ghidra_path = "/Users/matteotiozzo/Desktop/ghidra_11.1.2_PUBLIC"
    analyze_headless = os.path.join(ghidra_path, "support/analyzeHeadless")
    script_path = os.path.join(ghidra_path, "Ghidra/Features/Decompiler/ghidra_scripts")
    
    # Determina i nomi base del file
    file_base_name = os.path.basename(file_path)
    log_file_path = os.path.join(output_dir, f"{file_base_name}_ghidra.log")
    
    # Output per l'assembly di questo eseguibile
    assembly_output_path = os.path.join(output_dir, f"{file_base_name}_assembly.txt")

    # Costruisci il comando
    command = [
        analyze_headless,
        temp_project_path, "tempProject", "-import", file_path,
        "-postScript", "DecompileAllFunctions.py", assembly_output_path,
        "-scriptPath", script_path,
        "-processor", "x86:LE:64:default",
    ]

    # Esegui il comando
    try:
        subprocess.check_call(command)
        print(f"Decompilazione completata con successo per il file: {file_path}")
    except subprocess.CalledProcessError as e:
        print("Errore durante la decompilazione:", str(e))
        return

    # Verifica se il file assembly è stato creato correttamente
    if os.path.exists(assembly_output_path):
        print(f"File assembly salvato in: {assembly_output_path}")
    else:
        print(f"File assembly non trovato: {assembly_output_path}")


def process_all_files(directory, output_dir):
    assembly_dir = "/Users/matteotiozzo/Desktop/Thesis/code/Labeling/Assembly"
    for root, dirs, files in os.walk(directory):
        # Escludi la cartella "unrecognised"
        dirs[:] = [d for d in dirs if d != "unrecognised"]
        
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if os.path.isfile(file_path):
                # Verifica se esiste già un file .txt con lo stesso nome
                base_name = os.path.splitext(file_name)[0]
                txt_file_path = os.path.join(assembly_dir, f"{base_name}_assembly.txt")
                if os.path.exists(txt_file_path):
                    print(f"File assembly già esistente per: {file_path}")
                    continue
                
                # Crea un progetto Ghidra temporaneo per ciascun file
                with tempfile.TemporaryDirectory() as temp_project_path:
                    print(f"Processing file: {file_path}")
                    run_ghidra_decompilation(file_path, temp_project_path, output_dir)

if __name__ == "__main__":
    malware_directory = "/Users/matteotiozzo/Desktop/Thesis/code/Labeling/labels/"
    output_dir = "/Users/matteotiozzo/Desktop/Thesis/code/Labeling/Logs/"
    process_all_files(malware_directory, output_dir)
