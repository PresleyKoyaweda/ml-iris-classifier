#!/usr/bin/env python3
"""
Script d'entraînement du modèle de classification Iris
Entraîne un RandomForest et sauvegarde le modèle + métadonnées
"""

import json
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import mlflow

def main():
    print("=" * 60)
    print("🤖 ENTRAÎNEMENT DU MODÈLE IRIS CLASSIFIER")
    print("=" * 60)

    with mlflow.start_run():
        train(Path(__file__).parent / "outputs")

    print("\n" + "=" * 60)
    print("✅ ENTRAÎNEMENT TERMINÉ!")
    print("=" * 60)


def train(output_dir: Path) -> dict:
    """Entraîne le modèle et sauvegarde model.pkl + metadata.json dans output_dir.

    Retourne les métadonnées (utile pour les tests).
    """
    # ============================================================
    # 1. CHARGER LES DONNÉES
    # ============================================================
    print("\n📊 Étape 1: Chargement des données...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    print(f"   ✓ Dataset shape: {X.shape}")
    print(f"   ✓ Nombre de classes: {len(iris.target_names)}")
    print(f"   ✓ Classes: {', '.join(iris.target_names)}")
    
    # ============================================================
    # 2. SPLIT TRAIN/TEST
    # ============================================================
    print("\n📋 Étape 2: Split train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   ✓ Train set: {X_train.shape[0]} samples")
    print(f"   ✓ Test set: {X_test.shape[0]} samples")
    
    # ============================================================
    # 3. ENTRAÎNER LE MODÈLE
    # ============================================================
    print("\n🔧 Étape 3: Entraînement du modèle...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("   ✓ Modèle entraîné!")
    
    # ============================================================
    # 4. ÉVALUATION
    # ============================================================
    print("\n📈 Étape 4: Évaluation du modèle...")
    y_pred = model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted")
    }
    
    for metric_name, metric_value in metrics.items():
        print(f"   ✓ {metric_name}: {metric_value:.4f}")
    
    # ============================================================
    # 5. LOGS MLFLOW
    # ============================================================
    print("\n📝 Étape 5: Logging MLflow...")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("random_state", 42)
    
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)
    
    print("   ✓ Paramètres et métriques loggés")
    
    # ============================================================
    # 6. SAUVEGARDER LE MODÈLE
    # ============================================================
    print("\n💾 Étape 6: Sauvegarde du modèle...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / "model.pkl"
    joblib.dump(model, model_path)
    print(f"   ✓ Modèle sauvegardé: {model_path}")
    
    # ============================================================
    # 7. SAUVEGARDER LES MÉTADONNÉES
    # ============================================================
    print("\n📋 Étape 7: Sauvegarde des métadonnées...")
    metadata = {
        "model_name": "iris_classifier",
        "model_type": "RandomForestClassifier",
        "features": list(iris.feature_names),
        "target_classes": list(iris.target_names),
        "metrics": metrics,
        "params": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        }
    }
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✓ Métadonnées sauvegardées: {metadata_path}")
    
    print(f"\n📁 Fichiers générés:")
    print(f"   - {model_path}")
    print(f"   - {metadata_path}")
    print(f"\n🎯 Performance:")
    print(f"   - Accuracy:  {metrics['accuracy']:.2%}")
    print(f"   - F1-Score:  {metrics['f1']:.2%}")

    return metadata

if __name__ == "__main__":
    main()
