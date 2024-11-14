import pandas as pd
import os
from collections import defaultdict

# Directory for different families of malware
base_path = 'mnemonic_assembly_superfamily'
base_path = 'mnemonic_hex_superfamily'
output_base = 'output_mnemonics_hex'  # Directory to store output CSV files

# Ensure the output directory exists
if not os.path.exists(output_base):
    os.makedirs(output_base)

# Load CSV and extract all mnemonic pairs from column names
def load_csv_pairs(csv_path):
    df = pd.read_csv(csv_path)
    # Extract mnemonic pairs from all column names across all components, excluding the first column
    return sorted(set(tuple(sorted(col.split())) for col in df.columns[1:] if isinstance(col, str)))

# Process txt files and collect mnemonic pair occurrences for each file
def load_and_collect_txt_pairs(txt_path, csv_pairs):
    file_pairs = defaultdict(int)
    with open(txt_path, 'r', encoding='latin1') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                pair = tuple(sorted(parts[0].split()))
                occurrences = int(parts[1])
                if pair in csv_pairs:
                    file_pairs[pair] = occurrences
    return file_pairs

# Function to recursively find all files ending with "_mnemonics.txt" in a directory
def find_mnemonic_files(directory):
    mnemonic_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("_mnemonics.txt"):
                mnemonic_files.append(os.path.join(root, file))
    return mnemonic_files

# List of malware families
families = ['Adware', 'Backdoor', 'Downloader', 'Ransomware', 'Spyware', 'Trojan', 'Virus']

# Path to the coefficients file
csv_path = 'coefficients_0_001_rows_hex.csv'
csv_pairs = load_csv_pairs(csv_path)

# Convert csv_pairs to string format for column naming
csv_pairs_str = [' '.join(pair) for pair in csv_pairs]

# Process each family
for family in families:
    family_path = os.path.join(base_path, family)
    mnemonic_files = find_mnemonic_files(family_path)
    
    # Create an empty DataFrame with all csv_pairs as columns and index set to the malware filenames
    df = pd.DataFrame(columns=csv_pairs_str)
    
    # Process each TXT file and collect pairs
    for txt_path in mnemonic_files:
        # Get a unique identifier for each file based on its path relative to family directory
        file_name = os.path.relpath(txt_path, start=family_path).replace("\\", "/").replace("_mnemonics.txt", "")
        file_pairs = load_and_collect_txt_pairs(txt_path, csv_pairs)

        # Create a row for the current file, filling with occurrences or 0
        row_data = {(' '.join(pair)): file_pairs.get(pair, 0) for pair in csv_pairs}
        df.loc[file_name] = row_data  # Add the row to the DataFrame

    # Add 'Malware' as the first column and move the malware names from the index into this column
    df.insert(0, 'Malware', df.index)
    df.index = range(len(df))  # Reset the index to be numerical

    # Save the DataFrame to CSV
    output_csv_path = os.path.join(output_base, f'{family}_mnemonics_occurrences.csv')
    df.to_csv(output_csv_path, index=False, encoding='latin1')
