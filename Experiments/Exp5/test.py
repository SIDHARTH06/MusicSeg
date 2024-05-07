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
import librosa
import streamlit as st
import torch.nn.functional as F
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)
def display_audio_waveform(y, sr):
    st.subheader("Audio Waveform")
    st.write("This is the waveform representation of this  audio segment")
    st.write("----")
    # Plot waveform
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, len(y) / sr, len(y)), y)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Audio Waveform")
    st.pyplot()

# Function to display audio player
def display_audio_player(y, sr):
    st.subheader("Audio Player")
    st.audio(y, format="audio/wav",sample_rate = 22050)
def cut_audio_segments(audio_file, stop_values):
    # Load the audio file
    y, sr = librosa.load(audio_file,sr=22050)

    # Initialize list to store audio segments
    segments = []

    # Iterate over stop values to create segments
    for i in range(len(stop_values)):
        # Get start and stop timestamps for the current segment
        start = 0 if i == 0 else stop_values[i-1]
        stop = stop_values[i]

        # Convert timestamps to frame indices
        start_frame = librosa.time_to_samples(start, sr=22050)
        stop_frame = librosa.time_to_samples(stop, sr=22050)

        print(start_frame,stop_frame)
        # Extract audio segment
        segment = y[start_frame:stop_frame]

        # Append the segment to the list
        segments.append(segment)

    return segments
device = 'mps'
def remove_negative_and_sort(arr):
    # Filter out negative values
    positive_values = arr[arr >= 0]

    
    # Sort the array
    sorted_array = np.sort(positive_values)
    sorted_array = sorted_array / sorted_array[-1]
    
    return sorted_array
def pad_array(array):
        padded_array = np.pad(array, ((0, 13 - array.shape[0]), (0, 13 - array.shape[1])), mode='constant', constant_values=0)
        return padded_array
def extract_mfcc(y, sr=22050, n_mfcc=13, n_fft=2048, hop_length=512, duration=5):
        num_segments = int(len(y) / (duration * sr))
        mfccs = []
        for i in range(num_segments):
            segment = y[i * duration * sr : (i + 1) * duration * sr]
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfccs.append(mfcc_mean)
        return np.stack(mfccs, axis=0)
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
        x = self.fc3(x)
        return x

model = RegressionModel().to(device)
def process(audio_file):
    y, _ = librosa.load(audio_file, sr=22050)
    duration_sec = librosa.get_duration(y=y, sr=22050)
    y = extract_mfcc(y,duration=30)
    y= pad_array(y)
    y=y.reshape((1,1,13,13))
    x = model(torch.tensor(y,dtype=torch.float).to(device))
    return remove_negative_and_sort(x.detach().cpu().numpy()[0])*duration_sec


st.title('Structural Segmentation of Indian Pop Music')

audio_path = st.text_input("Enter the path to the audio file")

if st.button('Segment Audio') and audio_path:
    st.write("Segmenting audio...")
    res = process(audio_path)
    segments = cut_audio_segments(audio_path,res)
    for i,seg in enumerate(segments):
         st.write(f"Segment: {i}")
         display_audio_waveform(seg,22050)
         display_audio_player(seg,22050)



                

