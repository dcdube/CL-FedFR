import os
import pandas as pd

# Define the paths
base_path = r'D:\Research\Tubitak\Head_pose_split\Training_33-33-33\processed_with_sum_csv'
output_path = r'D:\Research\Tubitak\Head_pose_split\Training_33-33-33\processed_with_sum_csv\sumavg_joined.csv'

# List of client directories
client_directories = [f'cl_client_{str(i).zfill(4)}' for i in range(40)]

# Initialize an empty list to hold the DataFrames
dataframes = []

# Loop through each client directory
for client_dir in client_directories:
    csv_path = os.path.join(base_path, client_dir, f'sumavg_{client_dir}.csv')

    # Check if the CSV file exists
    if os.path.exists(csv_path):
        # Read the CSV file and add to the list
        data = pd.read_csv(csv_path)
        dataframes.append(data)
    else:
        print(f"CSV file not found for {client_dir}")

# Concatenate all DataFrames in the list
joined_data = pd.concat(dataframes, ignore_index=True)

# Save the joined data to a CSV file
joined_data.to_csv(output_path, index=False)
data = pd.read_csv(output_path)

# Add a new column "client_id" with values from 0 to 39
data.insert(0, 'client_id', range(1, 41))  # Assuming there are 40 clients
data.to_csv(output_path, index=False)
print("CSV files joined and saved.")
