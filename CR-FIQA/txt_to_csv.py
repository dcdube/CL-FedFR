# import csv
#
# # File paths
# input_file_path = '/home/master/FedFR-main/ms1m_split/split_pretrain_6000/client_0000/train/CRFIQAL_global.txt'
# output_file_path = '/home/master/FedFR-main/ms1m_split/split_pretrain_6000/client_0000/train/CRFIQAL_global.csv'
#
# # Read the input file and prepare data for CSV
# data = []
# with open(input_file_path, 'r') as file:
#     for idx, line in enumerate(file):
#         parts = line.strip().split()
#         crfiqa_score = parts[-1]
#         data.append([idx, crfiqa_score])
#
# # Write data to CSV file
# with open(output_file_path, 'w', newline='') as csvfile:
#     csv_writer = csv.writer(csvfile)
#     csv_writer.writerow(['identity', 'crfiqa_score'])  # Write header
#     csv_writer.writerows(data)
#
# print(f"CSV file saved to {output_file_path}")

# -----------------------------------------------------------------------------------------------------------------------
# GENERATE CSVs
# -----------------------------------------------------------------------------------------------------------------------

# import csv
# import os
#
# # Base directory
# base_dir = '/home/master/FedFR-main/ms1m_split/split_train_i4000c0040'
# client_count = 40
#
# # Initialize the starting identity value
# identity = 0
#
# # Loop through each client directory
# for i in range(client_count):
#     client_id = f"client_{i:04d}"
#     input_file_path = os.path.join(base_dir, f"{client_id}/train/CRFIQAL_{i:04d}.txt")
#     output_file_path = os.path.join(base_dir, f"{client_id}/train/CRFIQAL_{i:04d}.csv")
#
#     # Read the input file and prepare data for CSV
#     data = []
#     with open(input_file_path, 'r') as file:
#         for line in file:
#             parts = line.strip().split()
#             crfiqa_score = parts[-1]
#             data.append([identity, crfiqa_score])
#             identity += 1
#
#     # Write data to CSV file
#     with open(output_file_path, 'w', newline='') as csvfile:
#         csv_writer = csv.writer(csvfile)
#         csv_writer.writerow(['identity', 'crfiqa_score'])  # Write header
#         csv_writer.writerows(data)
#
#     print(f"CSV file saved to {output_file_path}")

# -----------------------------------------------------------------------------------------------------------------------
# JOIN CSVs
# -----------------------------------------------------------------------------------------------------------------------
#
import os
import pandas as pd

# Base directory
base_dir = '/home/master/FedFR-main/ms1m_split/split_train_i4000c0040'
output_file_path = os.path.join(base_dir, 'CRFIQA_local.csv')

# Initialize an empty list to hold data frames
df_list = []

# Loop through each client directory and read the CSV file
for i in range(40):
    client_id = f"client_{i:04d}"
    csv_file_path = os.path.join(base_dir, f"{client_id}/train/CRFIQAL_{i:04d}.csv")

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Append the DataFrame to the list
    df_list.append(df)

# Concatenate all DataFrames
combined_df = pd.concat(df_list, ignore_index=True)

# Save the combined DataFrame to a CSV file
combined_df.to_csv(output_file_path, index=False)

print(f"Combined CSV file saved to {output_file_path}")
