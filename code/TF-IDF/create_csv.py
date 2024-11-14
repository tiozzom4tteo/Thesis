import pandas as pd

def create_csv_from_principal_components(input_file, output_csv):
    # Initialize the storage for principal components
    pcs = {}
    current_pc = None

    # Read and process the file
    with open(input_file, 'r') as file:
        for line in file:
            if line.startswith('Principal Component'):
                # Start a new principal component
                current_pc = line.strip().split(':')[0]
                pcs[current_pc] = []
            elif current_pc and line.strip():
                # Add mnemonic pair, ignoring the coefficient for now
                mnemonic_pair = line.split(':')[0].strip()
                pcs[current_pc].append(mnemonic_pair)

    # Create DataFrame from the dictionary
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in pcs.items()])).T
    df.to_csv(output_csv, header=False, index=True)

# Path to the input text file
# input_path = 'sorted_principal_component_coefficients_filtered_0_001.txt'
input_path = 'sorted_principal_component_coefficients_filtered_0001_hex.txt'
# Path to save the output CSV file
# output_csv_path = 'coefficients_0_001_rows.csv'
output_csv_path = 'coefficients_0_001_rows_hex.csv'

# Run the function to process the file and create a CSV
create_csv_from_principal_components(input_path, output_csv_path)
