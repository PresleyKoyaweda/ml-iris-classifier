from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import json
from pathlib import Path
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Iris Classification API",
    description="API pour prédire la classe d'iris",
    version="1.0.0"
)

# Chemins vers le modèle et métadonnées
MODEL_PATH = Path("../train/outputs/model.pkl")
METADATA_PATH = Path("../train/outputs/metadata.json")

# Charger le modèle et métadonnées
try:
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    logger.info("✅ Modèle et métadonnées chargés")
except Exception as e:
    logger.error(f"❌ Erreur chargement modèle: {e}")
    model = None
    metadata = {}

# Schémas de requête/réponse
class IrisPredictionRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class IrisPredictionResponse(BaseModel):
    prediction: str
    probability: dict
    features: dict

# Routes
@app.get("/health")
async def health_check():
    """Vérifier que l'API fonctionne"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.get("/info")
async def get_model_info():
    """Obtenir les infos du modèle"""
    return {
        "model_name": metadata.get("model_name"),
        "model_type": metadata.get("model_type"),
        "features": metadata.get("features"),
        "target_classes": metadata.get("target_classes"),
        "metrics": metadata.get("metrics")
    }

@app.post("/predict", response_model=IrisPredictionResponse)
async def predict(request: IrisPredictionRequest):
    """
    Prédire la classe d'iris
    
    Exemple:
    {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Préparer les features
    features = np.array([
        request.sepal_length,
        request.sepal_width,
        request.petal_length,
        request.petal_width
    ]).reshape(1, -1)
    
    # Prédire
    prediction_idx = model.predict(features)[0]
    prediction_proba = model.predict_proba(features)[0]
    
    # Récupérer le nom de la classe
    target_classes = metadata.get("target_classes", ["setosa", "versicolor", "virginica"])
    prediction_class = target_classes[prediction_idx]
    
    # Formater les probabilités
    probability_dict = {
        target_classes[i]: float(prob) 
        for i, prob in enumerate(prediction_proba)
    }
    
    logger.info(f"Prédiction: {prediction_class} avec confiance {max(prediction_proba):.2%}")
    
    return IrisPredictionResponse(
        prediction=prediction_class,
        probability=probability_dict,
        features={
            "sepal_length": request.sepal_length,
            "sepal_width": request.sepal_width,
            "petal_length": request.petal_length,
            "petal_width": request.petal_width
        }
    )

@app.post("/predict-batch")
async def predict_batch(requests: list[IrisPredictionRequest]):
    """Prédire sur plusieurs exemples"""
    results = []
    for req in requests:
        result = await predict(req)
        results.append(result)
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
