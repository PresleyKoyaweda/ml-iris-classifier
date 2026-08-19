import json
import joblib
import numpy as np
from pathlib import Path

def init():
    global model
    model_path = Path(__file__).parent.parent / "train/outputs/model.pkl"
    model = joblib.load(model_path)
    print("Model loaded successfully")

def run(data):
    try:
        features = json.loads(data)
        X = np.array([features]).reshape(1, -1)
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        
        return {
            "prediction": int(prediction),
            "probabilities": proba.tolist()
        }
    except Exception as e:
        return {"error": str(e)}
