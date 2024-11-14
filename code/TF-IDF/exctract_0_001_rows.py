def process_principal_components(input_file, output_file):
    # Define constants for line processing
    PC_START_MARKER = 'Principal Component'
    MAX_LINES = 100
    THRESHOLD_PERCENTAGE = 0.001
    EXTRA_LINES = 20

    # Initialize variables
    pc_data = []
    current_pc = []
    line_count = 0

    with open(input_file, 'r') as file:
        for line in file:
            # Check for a new Principal Component section
            if PC_START_MARKER in line:
                # Process and filter the current PC data if it exists
                if current_pc:
                    # Calculate the max value for the current component
                    max_value = max(float(l.split()[-1]) for l in current_pc[1:])  # Exclude the title line
                    threshold = max_value * THRESHOLD_PERCENTAGE
                    # Filter lines based on the threshold
                    filtered_pc = [l for l in current_pc[1:] if float(l.split()[-1]) >= threshold]
                    
                    # Add a blank line and then additional 10 lines after the threshold if available
                    if len(filtered_pc) > 0:
                        last_index = current_pc.index(filtered_pc[-1])
                        filtered_pc.append('')  # Add a blank line
                        additional_lines = current_pc[last_index + 1:last_index + 1 + EXTRA_LINES]
                        filtered_pc.extend(additional_lines)
                    
                    # Append the title and the filtered lines
                    pc_data.append([current_pc[0]] + filtered_pc)

                # Start a new component
                current_pc = [line.strip()]
                line_count = 0
            elif current_pc is not None and line_count < MAX_LINES:
                # Collect up to MAX_LINES lines for the current PC
                current_pc.append(line.strip())
                line_count += 1

        # Process the last component after the loop ends
        if current_pc:
            max_value = max(float(l.split()[-1]) for l in current_pc[1:])
            threshold = max_value * THRESHOLD_PERCENTAGE
            filtered_pc = [l for l in current_pc[1:] if float(l.split()[-1]) >= threshold]
            if len(filtered_pc) > 0:
                last_index = current_pc.index(filtered_pc[-1])
                filtered_pc.append('')  # Add a blank line
                additional_lines = current_pc[last_index + 1:last_index + 1 + EXTRA_LINES]
                filtered_pc.extend(additional_lines)
            pc_data.append([current_pc[0]] + filtered_pc)

    # Write the collected data to the output file
    with open(output_file, 'w') as file:
        for pc in pc_data:
            for line in pc:
                file.write(line + '\n')
            file.write('\n')  # Add a blank line between PCs for clarity

# Specify the path to your input and output files
# input_path = 'sorted_principal_component_coefficients.txt'
input_path = 'sorted_principal_component_coefficients_hex.txt'

# output_path = 'sorted_principal_component_coefficients_filtered_0001.txt'
output_path = 'sorted_principal_component_coefficients_filtered_0001_hex.txt'

# Process the file
process_principal_components(input_path, output_path)
