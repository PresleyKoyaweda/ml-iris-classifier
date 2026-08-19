import json
import joblib
import numpy as np
from pathlib import Path

def init():
    global model
    model_path = Path(__file__).parent / "model.pkl"
    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    print("Model loaded successfully")

def run(data):
    try:
        if isinstance(data, str):
            data = json.loads(data)
        
        if isinstance(data, dict) and "data" in data:
            X = np.array(data["data"])
        else:
            X = np.array(data)
        
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        
        return {
            "prediction": int(prediction),
            "probabilities": proba.tolist()
        }
    except Exception as e:
        return {"error": str(e)}
