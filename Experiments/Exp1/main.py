import os
import json
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MusicDataset(Dataset):
    def __init__(self, audio_dir, json_dir, window_size=30):
        self.audio_dir = audio_dir
        self.json_dir = json_dir
        self.window_size = window_size
        self.audio_files = [file for file in os.listdir(audio_dir) if file.endswith(".mp3")]
        self.json_files = [file for file in os.listdir(json_dir) if file.endswith(".json")]

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file_path = os.path.join(self.audio_dir, self.audio_files[idx])
        y, sr = librosa.load(audio_file_path)

        json_file_path = os.path.join(self.json_dir, self.audio_files[idx].replace(".mp3", ".json"))
        with open(json_file_path, "r") as f:
            json_data = json.load(f)

        # Split audio into 30-second windows
        y_windows = librosa.effects.split(y, top_db=40)  # Adjust top_db as needed

        samples = []
        for window_start, window_end in y_windows:
            window_audio = y[window_start:window_end]

            # Extract start and stop timings from the JSON data
            start_times = [json_data["start"][str(i)] for i in range(len(json_data["start"])) if
                           window_start <= json_data["start"][str(i)] <= window_end]
            stop_times = [json_data["stop"][str(i)] for i in range(len(json_data["stop"])) if
                          window_start <= json_data["stop"][str(i)] <= window_end]

            # Pad start and stop times
            start_times = torch.nn.functional.pad(torch.FloatTensor(start_times), (0, 15 - len(start_times)))
            stop_times = torch.nn.functional.pad(torch.FloatTensor(stop_times), (0, 15 - len(stop_times)))

            # Extract MFCCs
            mfcc = librosa.feature.mfcc(y=window_audio, sr=sr, n_mfcc=13)
            mfcc = torch.nn.functional.pad(torch.FloatTensor(mfcc), (0, 100 - mfcc.shape[1]))

            # Extract Chroma
            chroma = librosa.feature.chroma_stft(y=window_audio, sr=sr)
            chroma = torch.nn.functional.pad(torch.FloatTensor(chroma), (0, 100 - chroma.shape[1]))

            # Extract Tempo
            onset_env = librosa.onset.onset_strength(y=window_audio, sr=sr)
            tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

            # Extract Rhythm Patterns (Example: using tempogram)
            tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
            rhythm_patterns = torch.nn.functional.pad(torch.FloatTensor(tempogram), (0, 100 - tempogram.shape[1]))

            sample = {
                'start_times': start_times,
                'stop_times': stop_times,
                'mfcc': mfcc,
                'chroma': chroma,
                'tempo': torch.FloatTensor([tempo]),  # Tempo as a single value
                'rhythm_patterns': rhythm_patterns
            }

            samples.append(sample)

        return samples

# Set your directory paths
audio_directory = "../../Dataset A/Raw/Songs"
json_directory = "../../Dataset A/Labels"

# Create dataset and dataloader
window_size = 30  # in seconds
music_dataset = MusicDataset(audio_directory, json_directory, window_size)
music_dataloader = DataLoader(music_dataset, batch_size=50, shuffle=True, collate_fn=lambda x: x)


import os
import librosa
import numpy as np
from pydub import AudioSegment
def load_audio(audio_path):
    try:
        audio = AudioSegment.from_file(audio_path)
        y = np.array(audio.get_array_of_samples())
        sr = audio.frame_rate
        return y, sr
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None, None
def extract_mfcc_features(audio_path, window_size=30):
    try:
        # Load audio file using librosa
        y, sr = load_audio(audio_path)
        print(y)

        # Calculate the number of frames for a 30-second window
        frame_size = int(window_size * sr)

        # Split the audio into windows of 30 seconds
        windows = librosa.util.frame(y, frame_length=frame_size, hop_length=frame_size)

        # Initialize an empty list to store MFCC features for each window
        mfcc_features = []

        # Extract MFCC features for each window
        for window in windows.T:
            mfcc = librosa.feature.mfcc(window, sr=sr)
            mfcc_features.append(mfcc)

        # Convert the list of MFCC features into a NumPy array
        mfcc_features = np.array(mfcc_features)

        return mfcc_features

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def process_and_save_folder(input_folder, output_folder, window_size=30):
    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp3"):
            # Construct the full path to the audio file
            audio_file_path = os.path.join(input_folder, filename)

            # Process and save the MFCC features for each audio file
            process_and_save(audio_file_path, output_folder, window_size)

def process_and_save(audio_file, output_folder, window_size=30):
    # Create a new folder in the drive with the same filename
    drive, filename = os.path.splitdrive(audio_file)
    filename_no_ext, _ = os.path.splitext(filename)
    output_folder_path = os.path.join(output_folder, filename_no_ext)
    os.makedirs(output_folder_path, exist_ok=True)

    # Extract MFCC features
    mfcc_features = extract_mfcc_features(audio_file, window_size)

    if mfcc_features is not None:
        # Save the MFCC features in a NumPy .npz file with the same filename
        npz_filename = os.path.join(output_folder_path, f"{filename_no_ext}_mfcc.npz")
        np.savez(npz_filename, mfcc_features=mfcc_features)
        print(f"MFCC features saved to {npz_filename}")

if __name__ == "__main__":
    # Specify the path to the folder containing audio files
    input_folder_path = "../../Dataset A/Raw/Songs"

    # Specify the window size in seconds
    window_size_seconds = 30

    # Specify the output folder
    output_folder_path = "./mfcc"

    # Process and save the MFCC features for all audio files in the input folder
    process_and_save_folder(input_folder_path, output_folder_path, window_size_seconds)

class AudioModel(nn.Module):
    def __init__(self):
        super(AudioModel, self).__init__()

        # Load pre-trained Hubert model
        # config = HubertConfig.from_pretrained("facebook/hubert-large-ll60k")
        # self.audio_encoder = HubertModel.from_pretrained("facebook/hubert-large-ll60k", config=config)

        self.feature_encoder = nn.Sequential(
            nn.Linear(in_features=100, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.conv1 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(16, 1, kernel_size=3, stride=1, padding=1)

        self.fc_start = nn.Linear(in_features=409, out_features=15)  # Adjusted output size to match ground truth
        self.fc_stop = nn.Linear(in_features=409, out_features=15)   # Adjusted output size to match ground truth

    def forward(self, mfcc, chroma, rhythm_patterns):
        # Combine audio features with other features
        combined_features = torch.cat([mfcc, chroma, rhythm_patterns], dim=1)

        # Encode combined features
        features_encoded = self.feature_encoder(combined_features)

        # Transpose to fit the expected shape for 1D convolution
        features_encoded = features_encoded.permute(0, 2, 1)

        # Apply 1D convolutions
        conv1_output = self.conv1(features_encoded)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)
        conv4_output = self.conv4(conv3_output)

        # Squeeze to remove the extra dimension added by the convolution
        conv_output = conv4_output.squeeze(dim=2)

        # Predict start and stop times
        start_times = self.fc_start(conv_output)
        stop_times = self.fc_stop(conv_output)

        return start_times, stop_times
    

from torch.optim.lr_scheduler import ExponentialLR
model = AudioModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ExponentialLR(optimizer, gamma=0.9)


model.to(device)
for epoch in range(50):
  loss=0.0
  counter=0
  for batch in tqdm(music_dataloader):
      # audio_data = pad_sequence([sample['audio'] for sample in batch], batch_first=True)
      mfcc = pad_sequence([sample['mfcc'] for sample in batch], batch_first=True)
      chroma = pad_sequence([sample['chroma'] for sample in batch], batch_first=True)
      rhythm_patterns = pad_sequence([sample['rhythm_patterns'] for sample in batch], batch_first=True)
      start_times = pad_sequence([sample['start_times'] for sample in batch], batch_first=True)
      stop_times = pad_sequence([sample['stop_times'] for sample in batch], batch_first=True)

      optimizer.zero_grad()

      start_pred, stop_pred = model(mfcc.to(device), chroma.to(device), rhythm_patterns.to(device))
      start_pred = start_pred.squeeze(dim=1)
      stop_pred = stop_pred.squeeze(dim=1)
      loss_start = criterion(start_pred.to(device), start_times.to(device))
      loss_stop = criterion(stop_pred.to(device), stop_times.to(device))

      total_loss = loss_start + loss_stop
      loss=total_loss.item()
      total_loss.backward()
      counter+=1
      optimizer.step()
      scheduler.step()

  print(f"Epoch {epoch} Loss: {loss/counter} Total Cumulative Loss: {total_loss.item()}")
