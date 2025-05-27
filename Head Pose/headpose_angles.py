import os
import csv

# Set the input and output file paths
input_folder = r'D:\Research\Tubitak\Head_pose_split\Pretrain_6000\pretrain_processed_with_sum_csv'
output_file = r'D:\Research\Tubitak\Head_pose_split\Pretrain_6000\pretrain_processed_with_sum_csv\headpose.csv'

# Initialize the joined data list
joined_data = []

# Loop through the folders and files
# for folder_num in range(40):
#     folder_name = f"cl_client_{folder_num:04}"
#     folder_path = os.path.join(input_folder, folder_name)
folder_path = input_folder
for file_num in range(6000):
    file_name = f"{file_num:06}.csv"
    file_path = os.path.join(folder_path, file_name)

    # Open the file and read its contents
    with open(file_path, "r") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row

        # Append the rows to the joined data list
        joined_data.extend(csv_reader)

# Write the joined data to the output file
with open(output_file, "w", newline="") as file:
    csv_writer = csv.writer(file)
    csv_writer.writerows(joined_data)
print(folder_path)

print("CSV files joined and saved as 'headpose.csv")
