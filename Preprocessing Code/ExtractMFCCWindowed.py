import os
import librosa
import numpy as np
INPUT_FOLDER = "/content/drive/MyDrive/Songs for annotation"
OUTPUT_FOLDER = "/content/drive/MyDrive/processed_mod_5"
def extract_mfcc(audio_file, sr=22050, n_mfcc=13, n_fft=2048, hop_length=512, duration=5):
    # Load audio file
    y, _ = librosa.load(audio_file, sr=sr)

    # Split the audio into segments of 30s each
    num_segments = int(len(y) / (duration * sr))
    mfccs = []
    for i in range(num_segments):
        segment = y[i * duration * sr : (i + 1) * duration * sr]
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc_mean = np.mean(mfcc, axis=1)  # Take the mean along the time axis
        mfccs.append(mfcc_mean)

    return mfccs

def process_folder(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each audio file in the input folder
    for file in os.listdir(input_folder):
        if file.endswith(".mp3"):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, os.path.splitext(file)[0] + ".npz")

            # Extract MFCC features
            mfccs = extract_mfcc(input_path)

            # Concatenate MFCCs for each song
            concatenated_mfccs = np.stack(mfccs, axis=0)
            print(concatenated_mfccs.shape)

            # Save the concatenated MFCCs in npz format
            np.savez(output_path, mfcc=concatenated_mfccs)

if __name__ == "__main__":
    input_folder = INPUT_FOLDER
    output_folder = OUTPUT_FOLDER
    process_folder(input_folder, output_folder)
