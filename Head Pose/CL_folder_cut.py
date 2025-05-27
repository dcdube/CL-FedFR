import os
import shutil

source_base_dir = r"D:\Research\Tubitak\Head_pose_split\Training_33-33-33\split_train_i4000c0040"
destination_base_dir = r"D:\Research\Tubitak\Head_pose_split\Training_33-33-33\split_train_i4000c0040\split_train_i4000c0040-25-25-25-25"

folders_to_move = ['easy', 'medium1', 'medium2', 'hard']
num_clients = 40

for client_num in range(num_clients):
    source_dir = os.path.join(source_base_dir, f"client_{client_num:04}")
    destination_dir = os.path.join(destination_base_dir, f"client_{client_num:04}")

    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Move the specified folders from the source to the destination
    for folder in folders_to_move:
        source_folder_path = os.path.join(source_dir, folder)
        destination_folder_path = os.path.join(destination_dir, folder)

        # Check if the source folder exists before moving
        if os.path.exists(source_folder_path):
            shutil.move(source_folder_path, destination_folder_path)
            print(f"Moved {folder} folder for client {client_num:04} to destination.")
        else:
            print(f"{folder} folder not found for client {client_num:04}.")

print("All folders moved successfully.")
