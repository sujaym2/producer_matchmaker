from fastapi import FastAPI, UploadFile, File
import uvicorn
from src.predict import predict_genre
from src.recommend import recommend_similar_tracks
import shutil
import os

UPLOAD_DIR = "../data/uploads/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    genre = predict_genre(file_path)
    return {"filename": file.filename, "predicted_genre": genre}

@app.post("/recommend/")
async def recommend(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    recommendations = recommend_similar_tracks(file_path)
    return {"filename": file.filename, "recommended_tracks": recommendations}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
