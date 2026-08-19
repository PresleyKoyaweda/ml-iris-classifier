import shutil
from pathlib import Path

import mlflow
import pytest

ROOT = Path(__file__).resolve().parent.parent
TRAIN_OUTPUTS = ROOT / "src" / "train" / "outputs"
API_MODEL_PATH = ROOT / "src" / "api" / "model.pkl"


@pytest.fixture(scope="session")
def trained_model():
    """Entraîne un modèle réel une fois par session de tests, à l'endroit
    où l'API et le script d'entraînement l'attendent (src/train/outputs)."""
    import train as train_module

    with mlflow.start_run():
        metadata = train_module.train(TRAIN_OUTPUTS)

    return {
        "output_dir": TRAIN_OUTPUTS,
        "model_path": TRAIN_OUTPUTS / "model.pkl",
        "metadata_path": TRAIN_OUTPUTS / "metadata.json",
        "metadata": metadata,
    }


@pytest.fixture(scope="session")
def scoring_model_copy(trained_model):
    """Copie le modèle entraîné à côté de score.py, comme le fait
    scripts/deploy_endpoint.py avant un déploiement Azure ML."""
    shutil.copy(trained_model["model_path"], API_MODEL_PATH)
    yield API_MODEL_PATH
    API_MODEL_PATH.unlink(missing_ok=True)
