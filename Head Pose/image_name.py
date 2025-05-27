import os
import csv

# Specify the base paths
image_base_path = r'D:\Research\Tubitak\Head_pose_split\Pretrain_6000\pretrain_images_6000'
csv_base_path = r'D:\Research\Tubitak\Head_pose_split\Pretrain_6000\pretrain_processed_with_sum_csv'

# # Iterate through the client folders
# for i in range(40):
#     client_folder = str(i).zfill(4)
#
#     # Generate the image folder path
#     image_folder_path = os.path.join(image_base_path, f'client_{client_folder}', 'cl_images')
#
#     # Generate the CSV folder path
#     # csv_folder_path = csv_base_path + client_folder
csv_folder_path = csv_base_path
image_folder_path = image_base_path

# Iterate through the CSV files in the folder
for j in range(6000):
    # Generate the CSV file path
    csv_file_path = os.path.join(csv_folder_path, str(j).zfill(6) + '.csv')

    # Check if the CSV file exists
    if os.path.isfile(csv_file_path):
        # Open the CSV file in read mode
        with open(csv_file_path, 'r') as file:
            # Read the existing rows
            reader = csv.reader(file)
            rows = list(reader)

        # Generate the image path for the current CSV file
        current_image_folder_path = os.path.join(image_folder_path, str(j).zfill(6))

        # Get the names of the image files in the current image folder
        image_names = [filename for filename in os.listdir(current_image_folder_path) if filename.endswith('.jpg')]

        # Ensure the number of image names matches the number of rows minus one in the CSV file
        if len(image_names) != len(rows) - 1:
            print(
                f"Number of image names ({len(image_names)}) does not match the number of rows minus one in {csv_file_path} ({len(rows) - 1}). Skipping...")
            continue

        # Add the 'image_name' header to the first row, first column
        rows[0].insert(0, 'image_name')

        # Iterate through the rows and add the image names
        for index, row in enumerate(rows[1:], start=1):
            row.insert(0, image_names[index - 1])

        # Open the CSV file in write mode and write the updated rows
        with open(csv_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)

        print(f"Image names added to {csv_file_path}")

# ----------------------------------------------------------------------------------------------------------------------

# import os
# import csv
#
# path1 = r"D:\Research\Tubitak\Datasets\processed_with_sum\cl_client_0000\000000.csv"
# path2 = r"C:\Users\deeve\Desktop\OpenFace\split_train_i4000c0040\client_0000\cl_images\000000"
#
# # Get a list of image files in path2
# image_files = [f for f in os.listdir(path2) if os.path.isfile(os.path.join(path2, f))]
#
# # Extract image names without file extensions
# image_names = [os.path.splitext(f)[0] for f in image_files]
#
# # Update the CSV file in path1
# rows = []
# with open(path1, mode='r') as csvfile:
#     reader = csv.reader(csvfile)
#     header = next(reader)  # Get the existing header
#     header.insert(0, "image_name")  # Add "images" as the first column header
#     rows.append(header)  # Add the updated header row
#     for i, row in enumerate(reader):
#         if i < len(image_names):
#             row.insert(0, image_names[i])  # Add image name to each row
#         rows.append(row)
#
# # Overwrite the CSV file with updated data
# with open(path1, mode='w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(rows)
#
# print("CSV file updated successfully.")
