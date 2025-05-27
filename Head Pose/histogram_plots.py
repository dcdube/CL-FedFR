import pandas as pd
import matplotlib.pyplot as plt

# Path to the CSV file
csv_file_path = r"D:\Research\Tubitak\Head_pose_split\Training_33-33-33\processed_with_sum-33-33-33\sumavg_joined.csv"

# Read the CSV file
data = pd.read_csv(csv_file_path)

# Extract data from columns
client_ids = data['client_id']
client_pose_sum = data['client_pose_sum']
client_pose_avg = data['client_pose_avg']

# Create histograms
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.bar(client_ids, client_pose_sum, color='blue', alpha=0.7)
plt.xlabel('Client ID')
plt.ylabel('Head Pose Angles Sum (°)')

plt.subplot(2, 1, 2)
plt.bar(client_ids, client_pose_avg, color='green', alpha=0.7)
plt.xlabel('Client ID')
plt.ylabel('Head Pose Angles Average (°)')

plt.tight_layout()
plt.show()
