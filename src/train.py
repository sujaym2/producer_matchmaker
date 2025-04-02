import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

DATA_FILE = "data/processed/features.csv"
MODEL_FILE = "models/xgboost_model.pkl"

df = pd.read_csv(DATA_FILE)

if "genre" not in df.columns:
    raise ValueError("You need to manually label some songs in features.csv before training.")

le = LabelEncoder()
df["genre_encoded"] = le.fit_transform(df["genre"])

X = df.drop(columns=["filename", "genre", "genre_encoded"])
y = df["genre_encoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

with open(MODEL_FILE, "wb") as f:
    pickle.dump((model, le), f)

print(f"Model saved to {MODEL_FILE}")