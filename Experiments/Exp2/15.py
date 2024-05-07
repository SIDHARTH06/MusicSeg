import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import re
from torch.utils.data import DataLoader, Dataset, random_split
device = 'cuda'
import matplotlib.pyplot as plt
class MusicDataset(Dataset):
    def __init__(self, mfcc_dir, json_dir):
        self.mfcc_dir = mfcc_dir
        self.json_dir = json_dir
        self.mfcc_files = [file for file in os.listdir(mfcc_dir) if file.endswith(".npz")]
        self.json_files = [file for file in os.listdir(json_dir) if file.endswith(".json")]
        self.csvfile = pd.read_csv('/workspace/Project_Final/MusicSeg/Dataset B/filename_mapping.csv')

    def __len__(self):
        return len(self.mfcc_files)

    def __getitem__(self, idx):
        mfcc_path = os.path.join(self.mfcc_dir, self.mfcc_files[idx])
        json_path = os.path.join(self.json_dir, self.mfcc_files[idx].replace(".npz", ".json"))

        # Load MFCC features
        mfcc_data = np.load(mfcc_path)['mfcc']
        mfcc_tensor = torch.tensor(mfcc_data, dtype=torch.float32)

        # Load JSON annotations
        annotations=pd.read_json(json_path)
        # Extract start and stop timestamps
        start_times = np.array(annotations['start'])
        stop_times = np.array(annotations['stop'])

        # Formulate target labels
        target_labels = []
        for i in range(26):  # 13 segments
            start_boundaries = np.where((start_times >= i * 15) & (start_times < (i + 1) * 15))[0]
            stop_boundaries = np.where((stop_times >= i * 15) & (stop_times < (i + 1) * 15))[0]
            if len(start_boundaries) > 0 or len(stop_boundaries) > 0:
                target_labels.append(1)
            else:
                target_labels.append(0)

        target_labels_tensor = torch.tensor(target_labels, dtype=torch.float32)

        return {'mfcc': mfcc_tensor, 'labels': target_labels_tensor}

# Example usage:
mfcc_dir = "../../Dataset B/Processed/MFCC/15"
json_dir = "../../Dataset B/Labels/"
dataset = MusicDataset(mfcc_dir, json_dir)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
print("testing dataloader..")
for batch in dataloader:
    mfcc = batch['mfcc']
    labels = batch['labels']
    print("MFCC Shape:", mfcc.shape)
    print("Labels:", labels)

print("..........")


train_size = int(0.7 * len(dataset))  # 70% of data for training
test_size = int(0.15 * len(dataset))  # 15% of data for testing
val_size = len(dataset) - train_size - test_size  # Remaining 15% for validation

# Use random_split to split the dataset
train_set, test_set, val_set = random_split(dataset, [train_size, test_size, val_size])


import torch
import torch.nn as nn
import torch.optim as optim
class MusicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MusicLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define LSTM units for each window
        self.lstm_layers = nn.ModuleList([nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
                                           for _ in range(output_size)])

        # Define linear layer to map LSTM output to label size
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        outputs = []
        for i in range(x.size(1)):
            lstm_out, _ = self.lstm_layers[i](x[:, i:i+1, :])
            output = self.linear(lstm_out[:, -1, :])  # Use only the last timestep output
            outputs.append(output)

        return torch.cat(outputs, dim=1)

# Example usage:
input_size = 13  # Assuming MFCC vector size is 13
hidden_size = 64
num_layers = 1
output_size = 26  # Number of LSTM units (one for each window)
model = MusicLSTM(input_size, hidden_size, num_layers, output_size)
print("Testing Model........")
# Test the model with random input
batch_size = 32
seq_length = 26
mfcc_tensor = torch.randn(batch_size, seq_length, input_size)  # Random input MFCC tensor
print(mfcc_tensor.shape)
output = model(mfcc_tensor)
print("Output shape:", output.shape)  # Should be batch_size x output_size

print("...................")


import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
import torch.optim as optim

# Assuming you have train_set, test_set, and val_set
train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

# Define your model, loss function, and optimizer
input_size = 13  # Assuming MFCC vector size is 13
hidden_size = 64
num_layers = 1
output_size = 26 # Number of LSTM units (one for each window)
model = MusicLSTM(input_size, hidden_size, num_layers, output_size)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, auc
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

# Initialize model, criterion, and optimizer
criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
model.to(device)
# Initialize lists to store evaluation metrics
f1_scores = []
precision_scores = []
recall_scores = []
pr_auc_values = []
precision_values=[]
recall_values=[]
# Open file to save result

with open('./result/res_15.txt', 'w') as f:

    # Perform 5-fold cross-validation
    model.train()
    for fold, (train_index, val_index) in enumerate(kf.split(train_set)):
        f.write(f"Fold [{fold+1}/5]")

        # Split data into training and validation sets for this fold
        train_subset = torch.utils.data.Subset(train_set, train_index)
        val_subset = torch.utils.data.Subset(train_set, val_index)
        train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)

        # Training loop for this fold
        num_epochs = 100
        model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in train_loader:
                mfcc = batch['mfcc'].to(device)
                labels = batch['labels'].to(device)
                # Forward pass
                outputs = model(mfcc)
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
        val_predictions = []
        val_targets = []
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                mfcc = batch['mfcc'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(mfcc)
                predicted_labels = torch.sigmoid(outputs) > 0.5
                val_predictions.extend(predicted_labels.cpu().numpy())
                labels = labels > 0.5
                val_targets.extend(labels.cpu().numpy())

        # Compute evaluation metrics for this fold
        model.eval()
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)
        f1 = f1_score(val_targets, val_predictions, average='weighted')
        precision = precision_score(val_targets, val_predictions, average='weighted')
        recall = recall_score(val_targets, val_predictions, average='weighted')
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)


        # Save validation scores to file

        f.write(f"\tValidation F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    # Compute metrics on the test set
    test_predictions = []
    test_targets = []
    model.eval()
    p_label=[]
    p_out=[]
    with torch.no_grad():
        for batch in test_loader:
            mfcc = batch['mfcc'].to(device)
            labels = batch['labels'] > 0.5
            outputs = model(mfcc)
            p_out.extend(outputs.detach().cpu().numpy())
            p_label.extend(labels.detach().cpu().numpy())
            predicted_labels = torch.sigmoid(outputs) > 0.5
            test_predictions.extend(predicted_labels.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

    test_predictions = np.array(test_predictions)
    test_targets = np.array(test_targets)
    test_f1 = f1_score(test_targets, test_predictions, average='weighted')
    test_precision = precision_score(test_targets, test_predictions, average='weighted')
    test_recall = recall_score(test_targets, test_predictions, average='weighted')

    f.write(f"\nTest F1 Score: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")

    # Save test scores to file

    p_out=np.array(p_out)
    p_label=np.array(p_label)
    thresholds = np.linspace(0, 1, 10)
    for threshold in thresholds:
        # Compute precision and recall for the current threshold
        predicted_labels = (p_out >= threshold).astype(int)
        precision = precision_score(p_label, predicted_labels, average='weighted')
        recall = recall_score(p_label, predicted_labels, average='weighted')
        precision_values.append(precision)
        recall_values.append(recall)
    plt.figure(figsize=(8, 6))
    plt.plot(recall_values, precision_values)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Different Thresholds')
    plt.grid(True)
    plt.show()
    # Save the figure
    plt.savefig(f'result/precision_recall_curve_15_{fold}.png')

    # Plot F1 scores, accuracies, and precisions in one plot
    plt.figure(figsize=(10, 5))

    # Plot F1 scores
    plt.plot(f1_scores, label='F1 Score', color='blue')

    # Plot precisions
    plt.plot(precision_scores, label='Precision', color='green')

    # Plot recalls
    plt.plot(recall_scores, label='Recall', color='red')

    plt.title('Evaluation Metrics vs Fold')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.grid(True)
    plt.legend()
    plt.savefig('result/evaluation_metrics_15.png')
    plt.show()

    # Save F1, precision, recall, and PR curve plots
    torch.save(model,'./models/model15.pt')