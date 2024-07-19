import json
import csv
import glob
import os

# Define the input files pattern and the output file
input_files_pattern = '../../harness-results/raw/*/LLM360__amber/samples_winogrande_*.jsonl'
output_file = '../../harness-results/winogrande_samples_combined.csv'

# Define the fields we are interested in
fields = ["sentence", "option1", "option2", "filtered_resps_0", "filtered_resps_1", "acc"]

# Find all files matching the pattern
input_files = glob.glob(input_files_pattern)

# Create a dictionary to hold all rows, keyed by sentence
all_rows = {}

# Dictionary to hold the folder names
folder_names = {}

# Helper function to find the group key for similar sentences
def get_group_key(sentence):
    return sentence[:50]  # Adjust this value if needed

# Read and process each file
for input_file in input_files:
    # Get the folder name from two levels higher
    folder_name = os.path.basename(os.path.dirname(os.path.dirname(input_file)))
    folder_names[input_file] = folder_name
    
    with open(input_file, 'r') as infile:
        for line in infile:
            # Parse each line as a JSON object
            data = json.loads(line)
            
            # Extract the required fields
            sentence = data["doc"]["sentence"]
            group_key = get_group_key(sentence)
            row = [
                data["doc"]["option1"],
                data["doc"]["option2"],
                data["filtered_resps"][0][0],
                data["filtered_resps"][1][0],
                data["acc"]
            ]
            
            if group_key not in all_rows:
                all_rows[group_key] = {}
                all_rows[group_key]['rows'] = {}
                all_rows[group_key]['sentence'] = sentence
            
            if folder_name not in all_rows[group_key]['rows']:
                all_rows[group_key]['rows'][folder_name] = row
            else:
                # If the entry already exists, sum the acc
                all_rows[group_key]['rows'][folder_name][4] += row[4]

# Write the combined rows to the CSV
with open(output_file, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    
    # Write the header row
    header = ["sentence"]
    for folder_name in set(folder_names.values()):
        header.extend([f"{field}_{folder_name}" for field in fields[1:]])  # Skip the 'sentence' field for other columns
    writer.writerow(header)
    
    # Write the data rows
    for group_key, group_data in all_rows.items():
        combined_row = [group_data['sentence']]
        for folder_name in set(folder_names.values()):
            combined_row.extend(group_data['rows'].get(folder_name, [''] * (len(fields) - 1)))
        writer.writerow(combined_row)
