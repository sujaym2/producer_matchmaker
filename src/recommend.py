import pandas as pd
from sklearn.neighbors import NearestNeighbors
from feature_extraction import extract_audio_features

DATA_FILE = "../data/processed/features.csv"

df = pd.read_csv(DATA_FILE)
X = df.drop(columns=["filename", "genre"])

nn_model = NearestNeighbors(n_neighbors=5, metric="euclidean")
nn_model.fit(X)

def recommend_similar_tracks(file_path):
    features = extract_audio_features(file_path)
    if features is None:
        return "Error extracting features"

    distances, indices = nn_model.kneighbors([features])
    recommended_tracks = df.iloc[indices[0]]["filename"].tolist()
    
    return recommended_tracks

test_file = "../data/raw/example_song.mp3"
recommended = recommend_similar_tracks(test_file)
print(f"Recommended tracks: {recommended}")
