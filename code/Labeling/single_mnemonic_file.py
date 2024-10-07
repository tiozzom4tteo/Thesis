# import os
# from collections import defaultdict
# import re

# # Path to the input folder
# input_dir = 'Assembly'

# # Output directory for mnemonic data files
# output_dir = 'Mnemonic_Assembly'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)  # Create the directory if it doesn't exist

# # Regex to extract mnemonics from a line of assembly code
# mnemonic_pattern = re.compile(r'\b[A-Z]+\b')

# # Process each file in the input folder
# for filename in os.listdir(input_dir):
#     if filename.endswith(".txt"):  # Check if the file is a .txt file
#         # Initialize dictionaries for each file
#         mnemonic_frequency = defaultdict(int)
#         mnemonic_files = defaultdict(set)

#         # Full path to the file
#         path = os.path.join(input_dir, filename)

#         with open(path, 'r') as file:
#             for line in file:
#                 mnemonics = mnemonic_pattern.findall(line)
#                 for mnemonic in set(mnemonics):  # Use set to ensure each mnemonic is counted once per line
#                     mnemonic_frequency[mnemonic] += 1
#                     mnemonic_files[mnemonic].add(filename)

#         # Create the CSV file for this specific assembly file in the output directory
#         filename = filename.replace("_assembly", "")  # Remove '_assembly' from the filename 
#         output_filename = f"{filename[:-4]}_mnemonics.csv"  # Remove '.txt' and append '_mnemonics.csv'
#         output_path = os.path.join(output_dir, output_filename)  # Full path to the output file

#         with open(output_path, 'w') as out_file:
#             # Write a header
#             out_file.write("Mnemonic  ,  Occurrences  \n")
#             # Write each mnemonic, its frequency, and the file count (always 1 in this context)
#             for mnemonic, count in sorted(mnemonic_frequency.items(), key=lambda item: item[1], reverse=True):
#                 file_count = len(mnemonic_files[mnemonic])  # Number of files the mnemonic appears in
#                 out_file.write(f"{mnemonic}  ,  {count}  \n")

#         # Output information for verification (optional)
#         print(f"Data for {filename} written to {output_path}")


import os
from collections import defaultdict
import re
from tqdm import tqdm

# Path to the input folder
input_dir = 'Assembly'

# Output directory for mnemonic data files
output_dir = 'Mnemonic_Assembly'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # Create the directory if it doesn't exist

# Regex to extract mnemonics from a line of assembly code
mnemonic_pattern = re.compile(r'\b[A-Z]+\b')

# List of files to process
files_to_process = [f for f in os.listdir(input_dir) if f.endswith(".txt")]

# Process each file in the input folder
for filename in tqdm(files_to_process, desc="Processing files"):
    # Initialize dictionaries for each file
    mnemonic_frequency = defaultdict(int)
    mnemonic_files = defaultdict(set)

    # Full path to the file
    path = os.path.join(input_dir, filename)

    with open(path, 'r') as file:
        for line in file:
            mnemonics = mnemonic_pattern.findall(line)
            for mnemonic in set(mnemonics):  # Use set to ensure each mnemonic is counted once per line
                mnemonic_frequency[mnemonic] += 1
                mnemonic_files[mnemonic].add(filename)

    # Create the CSV file for this specific assembly file in the output directory
    filename = filename.replace("_assembly", "")  # Remove '_assembly' from the filename 
    output_filename = f"{filename[:-4]}_mnemonics.csv"  # Remove '.txt' and append '_mnemonics.csv'
    output_path = os.path.join(output_dir, output_filename)  # Full path to the output file

    with open(output_path, 'w') as out_file:
        # Write a header
        out_file.write("Mnemonic  ,  Occurrences  \n")
        # Write each mnemonic, its frequency
        for mnemonic, count in sorted(mnemonic_frequency.items(), key=lambda item: item[1], reverse=True):
            out_file.write(f"{mnemonic}  ,  {count}  \n")

