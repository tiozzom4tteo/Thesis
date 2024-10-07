# import os
# from collections import defaultdict
# import re

# # Path to the input folder
# input_dir = 'Assembly'

# # Dictionary to count the frequency of mnemonics
# mnemonic_frequency = defaultdict(int)

# # Dictionary to map mnemonics to the files they appear in
# mnemonic_files = defaultdict(set)

# # Regex to extract mnemonics from a line of assembly code
# mnemonic_pattern = re.compile(r'\b[A-Z]+\b')

# # Process each file in the input folder
# for filename in os.listdir(input_dir):
#     if filename.endswith(".txt"):  # Ensure the file is a .txt file
#         path = os.path.join(input_dir, filename)
#         with open(path, 'r') as file:
#             for line in file:
#                 mnemonics = mnemonic_pattern.findall(line)
#                 for mnemonic in set(mnemonics):  # Use set to ensure each mnemonic is counted once per line
#                     mnemonic_frequency[mnemonic] += 1
#                     mnemonic_files[mnemonic].add(filename)

# # Create the file with the specified format
# with open('mnemonic_frequencies_all_files.csv', 'w') as file:
#     # Write a header
#     file.write("Mnemonic  ,  Occurrences  ,  File Count\n")
#     # Sort and write each mnemonic, its frequency, and the file count
#     for mnemonic, count in sorted(mnemonic_frequency.items(), key=lambda item: item[1], reverse=True):
#         file_count = len(mnemonic_files[mnemonic])  # Number of files the mnemonic appears in
#         file.write(f"{mnemonic}  ,  {count}  ,  {file_count}\n")

import os
from collections import defaultdict
import re
from tqdm import tqdm

# Path to the input folder
input_dir = 'Assembly'

# Dictionary to count the frequency of mnemonics
mnemonic_frequency = defaultdict(int)

# Dictionary to map mnemonics to the files they appear in
mnemonic_files = defaultdict(set)

# Regex to extract mnemonics from a line of assembly code
mnemonic_pattern = re.compile(r'\b[A-Z]+\b')

# Lista dei file da elaborare
files_to_process = [f for f in os.listdir(input_dir) if f.endswith(".txt")]

# Process each file in the input folder
for filename in tqdm(files_to_process, desc="Processing files"):
    path = os.path.join(input_dir, filename)
    with open(path, 'r') as file:
        for line in file:
            mnemonics = mnemonic_pattern.findall(line)
            for mnemonic in set(mnemonics):  # Use set to ensure each mnemonic is counted once per line
                mnemonic_frequency[mnemonic] += 1
                mnemonic_files[mnemonic].add(filename)

# Create the file with the specified format
with open('mnemonic_frequencies_all_files.csv', 'w') as file:
    # Write a header
    file.write("Mnemonic  ,  Occurrences  ,  File Count\n")
    # Sort and write each mnemonic, its frequency, and the file count
    for mnemonic, count in sorted(mnemonic_frequency.items(), key=lambda item: item[1], reverse=True):
        file_count = len(mnemonic_files[mnemonic])  # Number of files the mnemonic appears in
        file.write(f"{mnemonic}  ,  {count}  ,  {file_count}\n")
