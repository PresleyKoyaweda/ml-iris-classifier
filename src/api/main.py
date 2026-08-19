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

# Chemins FIXES (relatif au fichier, pas au cwd)
MODEL_PATH = Path(__file__).parent.parent / "train" / "outputs" / "model.pkl"
METADATA_PATH = Path(__file__).parent.parent / "train" / "outputs" / "metadata.json"

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

# Schémas
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
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.get("/info")
async def get_model_info():
    return {
        "model_name": metadata.get("model_name"),
        "model_type": metadata.get("model_type"),
        "features": metadata.get("features"),
        "target_classes": metadata.get("target_classes"),
        "metrics": metadata.get("metrics")
    }

@app.post("/predict", response_model=IrisPredictionResponse)
async def predict(request: IrisPredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        features = np.array([
            request.sepal_length,
            request.sepal_width,
            request.petal_length,
            request.petal_width
        ]).reshape(1, -1)
        
        prediction_idx = model.predict(features)[0]
        prediction_proba = model.predict_proba(features)[0]
        
        target_classes = metadata.get("target_classes", ["setosa", "versicolor", "virginica"])
        prediction_class = target_classes[prediction_idx]
        
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
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
