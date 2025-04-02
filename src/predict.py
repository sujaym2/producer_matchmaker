import pickle
import pandas as pd
from feature_extraction import extract_audio_features

MODEL_FILE = "../models/xgboost_model.pkl"

with open(MODEL_FILE, "rb") as f:
    model, le = pickle.load(f)

def predict_genre(file_path):
    features = extract_audio_features(file_path)
    if features is None:
        return "Error extracting features"
    
    features_df = pd.DataFrame([features], columns=[f"feature_{i}" for i in range(len(features))])
    predicted_label = model.predict(features_df)[0]
    predicted_genre = le.inverse_transform([predicted_label])[0]
    
    return predicted_genre

test_file = "../data/raw/example_song.mp3"
predicted_genre = predict_genre(test_file)
print(f"Predicted genre: {predicted_genre}")
