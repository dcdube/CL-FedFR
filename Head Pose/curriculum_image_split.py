import os
import csv
import shutil

# Define the source and destination directories
source_dir = r"D:\Research\Tubitak\Head_pose_split\Pretrain_6000\pretrain_processed_with_sum_csv"
image_dir = r"D:\Research\Tubitak\Head_pose_split\Pretrain_6000\pretrain_images_6000"

# # Iterate over the client directories
# for client_num in range(0, 40):
#     client_dir = "client_{:04d}".format(client_num)
#     csv_dir = os.path.join(source_dir, "cl_" + client_dir)
#     image_path = os.path.join(image_dir, client_dir, "cl_images")
csv_dir = source_dir
image_path = image_dir

# Iterate over the csv files in the current client directory
for file_num in range(6000):
    csv_filename = "{:06d}.csv".format(file_num)
    csv_file = os.path.join(csv_dir, csv_filename)

    # Read the csv file
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip the header row

        # Iterate over the rows in the csv file
        for row in reader:
            image_name = row[0]
            image_path_src = os.path.join(image_path, csv_filename[:-4], image_name)

            # Get the value from the last column
            value = float(row[-1])

            # Create the destination directories if they don't exist
            image_path_dst = os.path.join(image_path, csv_filename[:-4])
            os.makedirs(image_path_dst, exist_ok=True)

            easy_dir = os.path.join(image_path_dst, "easy")
            medium1_dir = os.path.join(image_path_dst, "medium1")
            medium2_dir = os.path.join(image_path_dst, "medium2")
            hard_dir = os.path.join(image_path_dst, "hard")
            os.makedirs(easy_dir, exist_ok=True)
            os.makedirs(medium1_dir, exist_ok=True)
            os.makedirs(medium2_dir, exist_ok=True)
            os.makedirs(hard_dir, exist_ok=True)

            # Copy the image to the respective folder based on the value
            if value < 8.20:
                shutil.copy2(image_path_src, easy_dir)
            if value < 12.84:
                shutil.copy2(image_path_src, medium1_dir)
            if value < 18.86:
                shutil.copy2(image_path_src, medium2_dir)
            shutil.copy2(image_path_src, hard_dir)
print(image_path)