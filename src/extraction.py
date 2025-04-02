import librosa
import numpy as np
import pandas as pd
import os

DATA_DIR = "data/raw"
OUTPUT_FILE = "data/processed/features.csv"

def extract_audio_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)  # Load audio
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)  # MFCCs
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)  # Chroma
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)  # Tempo
        return np.hstack([mfccs, chroma, tempo])  # Combine 
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

data = []
for filename in os.listdir(DATA_DIR):
    if filename.endswith(".mp3") or filename.endswith(".wav"):
        file_path = os.path.join(DATA_DIR, filename)
        features = extract_audio_features(file_path)
        if features is not None:
            data.append([filename] + list(features))

columns = ["filename"] + [f"feature_{i}" for i in range(len(data[0]) - 1)]
df = pd.DataFrame(data, columns=columns)

df.to_csv(OUTPUT_FILE, index=False)
print(f"Feature extraction complete. Saved to {OUTPUT_FILE}")
