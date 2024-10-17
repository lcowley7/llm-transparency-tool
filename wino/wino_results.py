import json
import csv
import glob
import os
import re

# Define the input files pattern and the output file
# input_files_pattern = '../../harness-results/raw/*/LLM360__amber/samples_winogrande_*.jsonl'
# output_file = '../../harness-results/winogrande_5shot_samples_combined.csv'

input_files_pattern = '../../harness-results/raw0shot/*/LLM360__amber/samples_winogrande_*.jsonl'
output_file = '../../harness-results/winogrande_0shot_samples_combined.csv'


# Find all files matching the pattern
input_files = glob.glob(input_files_pattern)

# Create a dictionary to hold all rows, keyed by sentence prefix
all_rows = {}

# Dictionary to hold the folder names
folder_names = {}

# Helper function to find the group key for similar sentences
def get_group_key(sentence):
    return sentence[:50]  # This captures the first 50 characters

# Read and process each file
for input_file in input_files:
    # Get the folder name from two levels higher
    folder_name = os.path.basename(os.path.dirname(os.path.dirname(input_file)))
    if folder_name == "main":
        folder_name = "ckpt_359"
    folder_names[input_file] = folder_name
    
    with open(input_file, 'r') as infile:
        for line in infile:
            # Parse each line as a JSON object
            data = json.loads(line)
            
            # Extract the required fields
            sentence = data["doc"]["sentence"]
            group_key = get_group_key(sentence)
            row = {
                "sentence": sentence,
                "option1": data["doc"]["option1"],
                "option2": data["doc"]["option2"],
                "filtered_resps_0": float(data["filtered_resps"][0][0]),
                "filtered_resps_1": float(data["filtered_resps"][1][0]),
                "acc": data["acc"]
            }
            
            if group_key not in all_rows:
                all_rows[group_key] = {
                    'sentences': [sentence],  # Initialize with the first sentence encountered
                    'option1': row['option1'],
                    'option2': row['option2'],
                    'rows': {folder_name: [row]}
                }
            else:
                if sentence not in all_rows[group_key]['sentences'] and len(all_rows[group_key]['sentences']) < 2:
                    all_rows[group_key]['sentences'].append(sentence)  # Ensure the second distinct sentence is added
                if folder_name not in all_rows[group_key]['rows']:
                    all_rows[group_key]['rows'][folder_name] = [row]
                else:
                    all_rows[group_key]['rows'][folder_name].append(row)

# Write the combined rows to the CSV
with open(output_file, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    
    # Write the header row
    header = ["sentence1", "sentence2", "option1", "option2", "has_pair", "longest_consecutive_2s"]
    for folder_name in sorted(folder_names.values(), key=lambda x: int(re.search(r'\d+', x).group() if re.search(r'\d+', x) else 0)):
        header.extend([f"filtered_resps_0_sent1_{folder_name}", f"filtered_resps_1_sent1_{folder_name}", f"resps_diff_sent1_{folder_name}"])
        header.extend([f"filtered_resps_0_sent2_{folder_name}", f"filtered_resps_1_sent2_{folder_name}", f"resps_diff_sent2_{folder_name}"])
        header.extend([f"acc_sum_{folder_name}"])
    writer.writerow(header)
    
    # Write the data rows
    for group_key, group_data in all_rows.items():
        sentences = group_data['sentences']
        has_pair = len(sentences) == 2
        sentence1 = sentences[0] if len(sentences) > 0 else ""
        sentence2 = sentences[1] if len(sentences) > 1 else ""

        combined_row = [sentence1, sentence2, group_data['option1'], group_data['option2'], has_pair]
        
        # current_longest_streak = 0
        current_streak = 0
        
        for folder_name in sorted(folder_names.values(), key=lambda x: int(re.search(r'\d+', x).group() if re.search(r'\d+', x) else 0)):
            folder_rows = group_data['rows'].get(folder_name, [])
            acc_sum = sum(row['acc'] for row in folder_rows)
                        
            row0 = folder_rows[0]
            combined_row.extend([row0['filtered_resps_0'], row0['filtered_resps_1'], row0['filtered_resps_0'] - row0['filtered_resps_1']])

            if (len(folder_rows) > 1):
                row1 = folder_rows[1]
                combined_row.extend([row1['filtered_resps_0'], row1['filtered_resps_1'], row1['filtered_resps_0'] - row1['filtered_resps_1']])
            else:
                # Extend dummy values when not a pair.
                combined_row.extend([0, 0, 0])                
            
            combined_row.extend([acc_sum])
                        
            # Track longest consecutive 2s at the end.
            if acc_sum == 2:
                current_streak += 1
            else:
                current_streak = 0
            # current_longest_streak = max(current_longest_streak, current_streak)

        combined_row.insert(5, current_streak)  # Insert right after 'has_pair'
        writer.writerow(combined_row)

# Print results
print(f"Number of pairs: {len([1 for data in all_rows.values() if len(data['sentences']) == 2])}")
print(f"Number of non-pair sentences: {len([1 for data in all_rows.values() if len(data['sentences']) == 1])}")
