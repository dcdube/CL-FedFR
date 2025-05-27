import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Path to the CSV file
csv_file_path = r"D:\Research\Tubitak\Head_pose_split\Training_33-33-33\processed_with_sum-33-33-33\headpose.csv"

# Read the CSV file
data = pd.read_csv(csv_file_path)

# Convert radians to degrees and take absolute values for pose angles
data['pose_Rx'] = np.degrees(np.abs(data['pose_Rx']))
data['pose_Ry'] = np.degrees(np.abs(data['pose_Ry']))
data['pose_Rz'] = np.degrees(np.abs(data['pose_Rz']))

# Extract data from columns
pose_Rx = data['pose_Rx']
pose_Ry = data['pose_Ry']
pose_Rz = data['pose_Rz']
sum_deg = data['sum_deg']

# Set the font size for tick labels
fontsz = 24.5

plt.rcParams['xtick.labelsize'] = fontsz
plt.rcParams['ytick.labelsize'] = fontsz
num_bars = 80

# Determine the bin width based on the data range
bin_width = (max(max(pose_Rx), max(pose_Ry), max(pose_Rz), max(sum_deg)) - min(min(pose_Rx), min(pose_Ry), min(pose_Rz), min(sum_deg))) / num_bars

# Create histograms
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.hist(pose_Rx, bins=np.arange(min(pose_Rx), max(pose_Rx) + bin_width, bin_width), color='blue', alpha=0.7)
plt.title('Histogram for Pitch Angles', fontsize=fontsz)
plt.xlabel('Angle (°)', fontsize=fontsz)
plt.ylabel('Frequency', fontsize=fontsz)
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))  # Set scientific notation
plt.grid(True)
plt.xlim([0, 100])  # Set x-axis range
plt.ylim([0, 150000])  # Set y-axis range

plt.subplot(2, 2, 2)
plt.hist(pose_Ry, bins=np.arange(min(pose_Ry), max(pose_Ry) + bin_width, bin_width), color='green', alpha=0.7)
plt.title('Histogram for Yaw Angles', fontsize=fontsz)
plt.xlabel('Angle (°)', fontsize=fontsz)
plt.ylabel('Frequency', fontsize=fontsz)
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))  # Set scientific notation
plt.grid(True)
plt.xlim([0, 100])  # Set x-axis range
plt.ylim([0, 100000])  # Set y-axis range

plt.subplot(2, 2, 3)
plt.hist(pose_Rz, bins=np.arange(min(pose_Rz), max(pose_Rz) + bin_width, bin_width), color='red', alpha=0.7)
plt.title('Histogram for Roll Angles', fontsize=fontsz)
plt.xlabel('Angle (°)', fontsize=fontsz)
plt.ylabel('Frequency', fontsize=fontsz)
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))  # Set scientific notation
plt.grid(True)
plt.xlim([0, 100])  # Set x-axis range
plt.ylim([0, 250000])  # Set y-axis range

plt.subplot(2, 2, 4)
plt.hist(sum_deg, bins=np.arange(min(sum_deg), max(sum_deg) + bin_width, bin_width), color='purple', alpha=0.7)
plt.title('Histogram for the Sum', fontsize=fontsz)
plt.xlabel('Angle (°)', fontsize=fontsz)
plt.ylabel('Frequency', fontsize=fontsz)
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))  # Set scientific notation
plt.grid(True)
plt.xlim([0, 270])  # Set x-axis range
plt.ylim([0, 100000])  # Set y-axis range

plt.tight_layout()
plt.show()








