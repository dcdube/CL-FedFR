import os
import shutil

base_path = r"D:\Research\Tubitak\Head_pose_split\Pretrain_6000\pretrain_images_6000"

# # Iterate over client folders
# for client_num in range(40):
#     client_folder = os.path.join(base_path, f"client_{client_num:04d}")
#     images_folder = os.path.join(client_folder, "cl_images")
#     # easy_folder = os.path.join(client_folder, "easy")
images_folder = base_path
client_folder = base_path
# Create "easy", "medium", and "hard" folders in each client folder
for difficulty in ["easy", "medium1", "medium2", "hard"]:
    difficulty_folder = os.path.join(client_folder, difficulty)
    os.makedirs(difficulty_folder, exist_ok=True)

# Iterate over easy subfolders in the "cl_images" folder
for subfolder in os.listdir(images_folder):
    subfolder_path = os.path.join(images_folder, subfolder)
    if os.path.isdir(subfolder_path):
        easy_folder_path = os.path.join(subfolder_path, "easy")
        new_folder_name = subfolder.zfill(6)  # Pad folder name with leading zeros

        # Move and rename the "easy" folder within each subfolder
        if os.path.exists(easy_folder_path):
            new_easy_folder_path = os.path.join(client_folder, "easy", new_folder_name)
            shutil.move(easy_folder_path, new_easy_folder_path)
    print(easy_folder_path)

# Iterate over medium subfolders in the "cl_images" folder
for subfolder in os.listdir(images_folder):
    subfolder_path = os.path.join(images_folder, subfolder)
    if os.path.isdir(subfolder_path):
        medium1_folder_path = os.path.join(subfolder_path, "medium1")
        new_folder_name = subfolder.zfill(6)  # Pad folder name with leading zeros

        # Move and rename the "easy" folder within each subfolder
        if os.path.exists(medium1_folder_path):
            new_medium_folder_path = os.path.join(client_folder, "medium1", new_folder_name)
            shutil.move(medium1_folder_path, new_medium_folder_path)
    print(medium1_folder_path)

# Iterate over medium subfolders in the "cl_images" folder
for subfolder in os.listdir(images_folder):
    subfolder_path = os.path.join(images_folder, subfolder)
    if os.path.isdir(subfolder_path):
        medium2_folder_path = os.path.join(subfolder_path, "medium2")
        new_folder_name = subfolder.zfill(6)  # Pad folder name with leading zeros

        # Move and rename the "easy" folder within each subfolder
        if os.path.exists(medium2_folder_path):
            new_medium_folder_path = os.path.join(client_folder, "medium2", new_folder_name)
            shutil.move(medium2_folder_path, new_medium_folder_path)
    print(medium2_folder_path)

# Iterate over hard subfolders in the "cl_images" folder
for subfolder in os.listdir(images_folder):
    subfolder_path = os.path.join(images_folder, subfolder)
    if os.path.isdir(subfolder_path):
        hard_folder_path = os.path.join(subfolder_path, "hard")
        new_folder_name = subfolder.zfill(6)  # Pad folder name with leading zeros

        # Move and rename the "easy" folder within each subfolder
        if os.path.exists(hard_folder_path):
            new_hard_folder_path = os.path.join(client_folder, "hard", new_folder_name)
            shutil.move(hard_folder_path, new_hard_folder_path)
    print(hard_folder_path)

print("Folders created and moved successfully.")
