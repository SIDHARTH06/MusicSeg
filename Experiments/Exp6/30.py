import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split

class MusicDataset(Dataset):
    def __init__(self, mfcc_dir, json_dir):
        self.mfcc_dir = mfcc_dir
        self.json_dir = json_dir
        self.mfcc_files = [file for file in os.listdir(mfcc_dir) if file.endswith(".npz")]
        self.json_files = [file for file in os.listdir(json_dir) if file.endswith(".json")]

    def __len__(self):
        return len(self.mfcc_files)

    def __getitem__(self, idx):
        mfcc_path = os.path.join(self.mfcc_dir, self.mfcc_files[idx])
        json_path = os.path.join(self.json_dir, self.mfcc_files[idx].replace(".npz", ".json"))

        # Load MFCC features
        mfcc_data = np.load(mfcc_path)['spectrogram']
        mfcc_tensor = torch.tensor(mfcc_data, dtype=torch.float32)

        # Load JSON annotations
        annotations=pd.read_json(json_path)

        # Extract start and stop timestamps
        start_times = np.array(annotations['start'])
        stop_times = np.array(annotations['stop'])

        # Formulate target labels
        target_labels = []
        for i in range(13):  # 13 segments
            start_boundaries = np.where((start_times >= i * 30) & (start_times < (i + 1) * 30))[0]
            stop_boundaries = np.where((stop_times >= i * 30) & (stop_times < (i + 1) * 30))[0]
            if len(start_boundaries) > 0 or len(stop_boundaries) > 0:
                target_labels.append(1)
            else:
                target_labels.append(0)

        target_labels_tensor = torch.tensor(target_labels, dtype=torch.float32)

        return {'mfcc': mfcc_tensor, 'labels': target_labels_tensor}

# Example usage:
mfcc_dir = "../../Dataset A/Processed/Spectrogram/30swindow"
json_dir = "../../Dataset A/Labels"
dataset = MusicDataset(mfcc_dir, json_dir)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

for batch in dataloader:
    mfcc = batch['mfcc']
    labels = batch['labels']
    print("Spectrogram Shape:", mfcc.shape)
    print("Labels:", labels)


train_size = int(0.7 * len(dataset))  # 70% of data for training
test_size = int(0.15 * len(dataset))  # 15% of data for testing
val_size = len(dataset) - train_size - test_size  # Remaining 15% for validation

# Use random_split to split the dataset
train_set, test_set, val_set = random_split(dataset, [train_size, test_size, val_size])

import torch
import torch.nn as nn

class CustomResNet(nn.Module):
    def __init__(self, input_channels=1292, num_classes=2):
        super(CustomResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold

# Define your model, loss function, and optimizer
input_size = 128  # Number of frequency bins in the spectrogram
hidden_size = 64
num_layers = 1
output_size = 13  # Number of windows
model = CustomResNet(input_size, hidden_size, num_layers, output_size)  # Assuming you've defined CustomResNet
criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store evaluation metrics
f1_scores = []
precision_scores = []
recall_scores = []

# Perform 5-fold cross-validation
for fold, (train_index, val_index) in enumerate(kf.split(train_set)):
    print(f"Fold [{fold+1}/5]")

    # Split data into training and validation sets for this fold
    train_subset = torch.utils.data.Subset(train_set, train_index)
    val_subset = torch.utils.data.Subset(train_set, val_index)
    train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)

    # Training loop for this fold
    num_epochs = 100
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            spectrogram, labels = batch
            # Forward pass
            outputs = model(spectrogram)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print average loss for the epoch
        average_loss = total_loss / len(train_loader)
        print(f"\tEpoch [{epoch+1}/{num_epochs}], Average Loss: {average_loss:.4f}")

    # Evaluation on validation set
    model.eval()
    val_predictions = []
    val_targets = []
    with torch.no_grad():
        for batch in val_loader:
            spectrogram, labels = batch
            outputs = model(spectrogram)
            predicted_labels = torch.sigmoid(outputs) > 0.5
            val_predictions.extend(predicted_labels.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())

    # Compute evaluation metrics for this fold
    val_predictions = np.array(val_predictions)
    val_targets = np.array(val_targets)
    f1 = f1_score(val_targets, val_predictions, average='weighted')
    precision = precision_score(val_targets, val_predictions, average='weighted')
    recall = recall_score(val_targets, val_predictions, average='weighted')
    f1_scores.append(f1)
    precision_scores.append(precision)
    recall_scores.append(recall)

    print(f"\tValidation F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

# Compute metrics on the test set
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
test_predictions = []
test_targets = []
with torch.no_grad():
    for batch in test_loader:
        spectrogram, labels = batch
        outputs = model(spectrogram)
        predicted_labels = torch.sigmoid(outputs) > 0.5
        test_predictions.extend(predicted_labels.cpu().numpy())
        test_targets.extend(labels.cpu().numpy())

test_predictions = np.array(test_predictions)
test_targets = np.array(test_targets)
test_f1 = f1_score(test_targets, test_predictions, average='weighted')
test_precision = precision_score(test_targets, test_predictions, average='weighted')
test_recall = recall_score(test_targets, test_predictions, average='weighted')

print(f"\nTest F1 Score: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
