import csv
import os
import math
from math import degrees

# Define the directory path
base_dir = r'C:\Users\deeve\Desktop\OpenFace\processed'
destination_dir = r'D:\Research\Tubitak\Head_pose_split\Training_33-33-33\processed_with_sum_csv'

# Iterate over each client folder
for client_num in range(40):
    client_folder = f'cl_client_{client_num:04d}'

        # Iterate over each CSV file
    for file_num in range(100):
        file_name = f'{file_num:06d}.csv'
        print(file_name)
        file_path = os.path.join(base_dir, client_folder, file_name)
        # destination_file_path = os.path.join(destination_dir, client_folder, file_name)
        destination_file_path = os.path.join(destination_dir, client_folder, file_name)

        # Check if the file exists
        if os.path.isfile(file_path):
            # Read the CSV file
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                rows = list(reader)

            # Add new column headers
            rows[0].extend(['sum_rad', 'sum_deg'])

            # Calculate and add values to each row
            for i in range(1, len(rows)):
                pose_Rx = float(rows[i][8])
                pose_Ry = float(rows[i][9])
                pose_Rz = float(rows[i][10])
                sum_rad = abs(pose_Rx) + abs(pose_Ry) + abs(pose_Rz)
                sum_deg = degrees(sum_rad)
                rows[i].extend([round(sum_rad, 3), round(sum_deg, 2)])

            # Write the updated data back to the CSV file
            os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)
            with open(destination_file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(rows)

# ----------------------------------------------------------------------------------------------------------------------
# import csv
# import os
#
# # Define the directory path
# base_dir = r'C:\Users\deeve\Desktop\OpenFace\processed'
#
# # Iterate over each client folder
# for client_num in range(40):
#     client_folder = f'cl_client_{client_num:04d}'
#
#     # Iterate over each CSV file
#     for file_num in range(100):
#         file_name = f'{file_num:06d}.csv'
#         file_path = os.path.join(base_dir, client_folder, file_name)
#
#         # Check if the file exists
#         if os.path.isfile(file_path):
#             # Read the CSV file
#             with open(file_path, 'r') as file:
#                 reader = csv.reader(file)
#                 rows = list(reader)
#
#             # Delete the last 4 columns from each row
#             for i in range(len(rows)):
#                 rows[i] = rows[i][:-4]
#
#             # Write the updated data back to the CSV file
#             with open(file_path, 'w', newline='') as file:
#                 writer = csv.writer(file)
#                 writer.writerows(rows)
