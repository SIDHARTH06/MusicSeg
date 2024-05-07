import os
import numpy as np

# Directory containing the npz files
folder_path = "/workspace/Project_Final/MusicSeg/Dataset B/Processed/MFCC/1"

# List all files in the directory
files = os.listdir(folder_path)

# Find the maximum size among all arrays
max_size = None
for file in files:
    if file.endswith(".npz"):
        data = np.load(os.path.join(folder_path, file))
        for key in data.keys():
            if max_size is None:
                max_size = data[key].shape
            else:
                max_size = tuple(max(max_size[i], data[key].shape[i]) for i in range(len(max_size)))
print(max_size)
# Pad each array with zeros to match the maximum size
for file in files:
    if file.endswith(".npz"):
        data = np.load(os.path.join(folder_path, file))
        padded_data = {}
        for key in data.keys():
            padding = [(0, max_size[i] - data[key].shape[i]) for i in range(len(max_size))]
            padded_data[key] = np.pad(data[key], padding, mode='constant')

        # Save the padded data back to the npz file
        np.savez_compressed(os.path.join(folder_path, file), **padded_data)
