import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        # Compute binary cross entropy
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute the exponential term
        pt = torch.exp(-BCE_loss)

        # Compute the focal loss
        focal_loss = (1 - pt) ** self.gamma * BCE_loss

        # Apply class balancing weights
        focal_loss = self.alpha * focal_loss

        # Average the loss
        return torch.mean(focal_loss)

# Example usage:
# Assuming your model outputs logits, and targets are one-hot encoded labels
criterion = FocalLoss(gamma=2, alpha=0.25)
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
        mfcc_data = np.load(mfcc_path)['mfcc']
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
mfcc_dir = "../../Dataset A/Processed/MFCC/30swindow"
json_dir = "../../Dataset A/Labels"
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

        # Define linear layers to map LSTM output to single sigmoid output
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(output_size)])

    def forward(self, x):
        outputs = []
        for i in range(x.size(1)):
            lstm_out, _ = self.lstm_layers[i](x[:, i:i+1, :])
            linear_out = self.linear_layers[i](lstm_out[:, -1, :])  # Apply linear layer to LSTM output
            sigmoid_out = torch.sigmoid(linear_out)  # Apply sigmoid activation
            outputs.append(sigmoid_out)

        return torch.cat(outputs, dim=1)

# Example usage:
input_size = 13  # Assuming MFCC vector size is 13
hidden_size = 64
num_layers = 1
output_size = 13  # Number of LSTM units (one for each window)
model = MusicLSTM(input_size, hidden_size, num_layers, output_size)
print("Testing Model........")
# Test the model with random input
batch_size = 32
seq_length = 13
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

input_size = 13  # Assuming MFCC vector size is 13
hidden_size = 64
num_layers = 1
output_size = 13 # Number of LSTM units (one for each window)
model = MusicLSTM(input_size, hidden_size, num_layers, output_size)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, auc
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
# criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
model.to(device)
# Initialize lists to store evaluation metrics
f1_scores = []
precision_scores = []
recall_scores = []
pr_auc_values = []

# Open file to save results
with open('results/res_30.txt', 'w') as f:
    f.write("Fold\tF1 Score\tPrecision\tRecall\n")

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
    model.eval()
    val_predictions = []
    val_targets = []
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
    val_predictions = np.array(val_predictions)
    val_targets = np.array(val_targets)
    f1 = f1_score(val_targets, val_predictions, average='weighted')
    precision = precision_score(val_targets, val_predictions, average='weighted')
    recall = recall_score(val_targets, val_predictions, average='weighted')
    f1_scores.append(f1)
    precision_scores.append(precision)
    recall_scores.append(recall)

    # Compute precision-recall curve for this fold
    precision_fold, recall_fold, _ = precision_recall_curve(val_targets, val_predictions)
    pr_auc_fold = auc(recall_fold, precision_fold)
    pr_auc_values.append(pr_auc_fold)

    # Save validation scores to file
    with open('results/res_30.txt', 'a') as f:
        f.write(f"{fold+1}\t{f1:.4f}\t{precision:.4f}\t{recall:.4f}\n")

    print(f"\tValidation F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, PR AUC: {pr_auc_fold:.4f}")

# Compute metrics on the test set
test_predictions = []
test_targets = []
with torch.no_grad():
    for batch in test_loader:
        mfcc = batch['mfcc'].to(device)
        labels = batch['labels'] > 0.5
        outputs = model(mfcc)
        predicted_labels = torch.sigmoid(outputs) > 0.5
        test_predictions.extend(predicted_labels.cpu().numpy())
        test_targets.extend(labels.cpu().numpy())

test_predictions = np.array(test_predictions)
test_targets = np.array(test_targets)
test_f1 = f1_score(test_targets, test_predictions, average='weighted')
test_precision = precision_score(test_targets, test_predictions, average='weighted')
test_recall = recall_score(test_targets, test_predictions, average='weighted')

print(f"\nTest F1 Score: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")

# Save test scores to file
with open('results/res_30.txt', 'a') as f:
    f.write("Test Scores:\n")
    f.write(f"F1 Score: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}\n")

print("plotting")

# Plotting Precision-Recall curve
plt.figure(figsize=(10, 5))

for fold in range(len(pr_auc_values)):
    plt.plot(recall_values[fold], precision_values[fold], label=f'Fold {fold+1} (AUC = {pr_auc_values[fold]:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid(True)
plt.savefig('results/precision_recall_curve_30.png')
plt.show()

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
plt.savefig('results/evaluation_metrics_30.png')
plt.show()

# Save PR curve data
np.save('results/precision_values_30.npy', np.array(precision_values))
np.save('results/recall_values_30.npy', np.array(recall_values))
np.save('results/pr_auc_values_30.npy', np.array(pr_auc_values))

# Save F1, precision, recall, and PR curve plots
torch.save(model,'./models/model30.pt')