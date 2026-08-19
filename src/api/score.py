import json
import os
import sys
import joblib
import numpy as np

def init():
    global model
    try:
        # Chercher le modèle dans plusieurs emplacements
        model_paths = [
            "/var/azureml-app/mnt/batch/outputs/model.pkl",
            "../../train/outputs/model.pkl",
            "/home/azureuser/models/model.pkl",
        ]
        
        model = None
        for path in model_paths:
            if os.path.exists(path):
                print(f"Loading model from {path}")
                model = joblib.load(path)
                break
        
        if model is None:
            print("WARNING: Model not found in expected paths")
            print(f"Current directory: {os.getcwd()}")
            print(f"Directory contents: {os.listdir('.')}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def run(data):
    try:
        if isinstance(data, str):
            data = json.loads(data)
        
        # Convertir en array numpy
        if isinstance(data, dict):
            # Format: {"data": [[5.1, 3.5, 1.4, 0.2]]}
            if "data" in data:
                X = np.array(data["data"])
            else:
                # Format: {"sepal_length": 5.1, ...}
                X = np.array([[data.get("sepal_length", 0),
                             data.get("sepal_width", 0),
                             data.get("petal_length", 0),
                             data.get("petal_width", 0)]])
        else:
            X = np.array(data)
        
        # Prédire
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        
        return {
            "prediction": int(prediction),
            "probabilities": proba.tolist(),
            "class_names": ["setosa", "versicolor", "virginica"]
        }
    except Exception as e:
        return {"error": str(e)}
