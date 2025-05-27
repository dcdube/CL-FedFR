import pandas as pd
import os
import csv
from math import degrees

base_directory = r'D:\Research\Tubitak\Head_pose_split\Training_33-33-33\processed_with_sum_csv'

for client_num in range(40):  # Iterate through clients from 0000 to 0039
    client_folder = f'cl_client_{str(client_num).zfill(4)}'
    directory_path = os.path.join(base_directory, client_folder)

    # List to store individual dataframes
    dataframes = []

    # Loop through the range of file numbers (000000 to 000099)
    for i in range(100):
        file_number = str(i).zfill(6)  # Convert index to zero-padded string
        file_path = os.path.join(directory_path, file_number + '.csv')

        # Check if the file exists before trying to read it
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            dataframes.append(df)

    # Concatenate dataframes along rows
    final_df = pd.concat(dataframes, ignore_index=True)

    # Path to save the joined CSV file
    output_file_path = os.path.join(directory_path, f'{client_folder}.csv')

    # Save the final concatenated dataframe as a CSV file
    final_df.to_csv(output_file_path, index=False)

    # # Check if the file exists
    # if os.path.isfile(output_file_path):
    #     # Read the CSV file
    #     with open(output_file_path, 'r') as file:
    #         reader = csv.reader(file)
    #         rows = list(reader)
    #
    #     # Add new column headers
    #     rows[0].extend(['sum_rad', 'sum_deg'])
    #
    #     # Calculate and add values to each row
    #     for i in range(1, len(rows)):
    #         pose_Rx = float(rows[i][8])
    #         pose_Ry = float(rows[i][9])
    #         pose_Rz = float(rows[i][10])
    #         sum_rad = abs(pose_Rx) + abs(pose_Ry) + abs(pose_Rz)
    #         sum_deg = degrees(sum_rad)
    #         rows[i].extend([round(sum_rad, 3), round(sum_deg, 2)])
    #
    #     # Write the updated data back to the CSV file
    #     os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    #     with open(output_file_path, 'w', newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerows(rows)

    df = pd.read_csv(output_file_path)
    client_pose_sum = round(df["sum_deg"].sum(), 2)
    client_pose_avg = round(df["sum_deg"].mean(), 2)
    result_df = pd.DataFrame(columns=["client_pose_sum", "client_pose_avg"])
    result_df.loc[0] = [client_pose_sum, client_pose_avg]
    sum_avg_file_path = os.path.join(directory_path, f'sumavg_{client_folder}.csv')
    result_df.to_csv(sum_avg_file_path, index=False)

    # Print success message for each client
    print(f"CSV files joined and saved for {client_folder}.")

print("All CSV files joined and saved successfully.")
