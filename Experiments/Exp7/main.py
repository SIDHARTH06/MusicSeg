import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import re
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
device = 'mps'
class MusicDataset(Dataset):
    def __init__(self, mfcc_dir, json_dir):
        self.mfcc_dir = mfcc_dir
        self.json_dir = json_dir
        self.mfcc_files = [file for file in os.listdir(mfcc_dir) if file.endswith(".npz")]
        self.json_files = [file for file in os.listdir(json_dir) if file.endswith(".json")]
        self.csvfile = pd.read_csv('../../Dataset B/filename_mapping.csv')

    def __len__(self):
        return len(self.mfcc_files)

    def __getitem__(self, idx):
        mfcc_path = os.path.join(self.mfcc_dir, self.mfcc_files[idx])
        json_path = os.path.join(self.json_dir, self.mfcc_files[idx].replace(".npz", ".json"))

        # Load MFCC features
        mfcc_data = np.load(mfcc_path)['mfcc']
        mfcc_data=mfcc_data.reshape((1,13,13))
        mfcc_tensor = torch.tensor(mfcc_data, dtype=torch.float32)


        # Load JSON annotations
        annotations=pd.read_json(json_path)
        # Extract start and stop timestamps
        stop_times = np.array(annotations['stop'])
        max_val = np.max(stop_times)
        normalized_array = (stop_times) / (max_val)
        padded_start = np.pad(normalized_array, (0, 25 - len(stop_times)), mode='constant', constant_values=-1)
        
        target_labels_tensor = torch.tensor(padded_start, dtype=torch.float32)
        

        return {'mfcc': mfcc_tensor, 'labels': target_labels_tensor}

# Example usage:
mfcc_dir = "../../Dataset B/Processed/MFCC/30"
json_dir = "../../Dataset B/Labels/"
dataset = MusicDataset(mfcc_dir, json_dir)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
print("testing dataloader..")
for batch in dataloader:
    mfcc = batch['mfcc']
    labels = batch['labels']
    print("MFCC Shape:", mfcc.shape)
    print("Labels:", labels[0].shape)

print("........")

import torch
import torch.nn as nn
import torch.nn.functional as F

class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64, 128)  # Adjusted based on input size
        self.fc2 = nn.Linear(128, 50)
        self.fc3 = nn.Linear(50, 25)  # Output size adjusted to 25
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        return x

# Instantiate the model
model = RegressionModel()
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
model.to(device)
def process_vector(vector):
    # Change all negative values to zero
    vector[vector < 0] = 0
    # Multiply all values by 250
    vector *= 250
    return vector
def filter_out_zeros(vector):
    # Filter out zero values
    non_zero_indices = vector != 0
    filtered_vector = vector[non_zero_indices]
    return filtered_vector
# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch in dataloader:
        mfcc = batch['mfcc'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(mfcc)  # add a channel dimension
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss}")

torch.save(model,'models/model.pt')

