import os
from pathlib import Path
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, Model, Environment
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv("../.env")

workspace_name = os.getenv("AZURE_ML_WORKSPACE_NAME")
resource_group = os.getenv("AZURE_ML_RESOURCE_GROUP")
subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name
)

# Créer l'environnement avec conda.yml
print("🔧 Création de l'environnement...")
env = Environment(
    name="iris-env",
    description="Environment for Iris classifier",
    conda_file="src/train/conda.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
)
registered_env = ml_client.environments.create_or_update(env)
print(f"✅ Environnement créé: {registered_env.name}:{registered_env.version}")

# Enregistrer le modèle
print("📦 Enregistrement du modèle...")
model_path = Path("src/train/outputs/model.pkl")
model = Model(
    path=str(model_path),
    name="iris-classifier",
    description="Iris Classification Model",
    type="custom_model"
)
registered_model = ml_client.models.create_or_update(model)
print(f"✅ Modèle enregistré: {registered_model.name}:{registered_model.version}")

# Créer l'endpoint
print("🔧 Création de l'endpoint...")
endpoint = ManagedOnlineEndpoint(
    name="iris-classifier",
    description="Iris Classification Endpoint"
)

try:
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print("✅ Endpoint créé/mis à jour")
except:
    print("⚠️ Endpoint existe déjà")

# Créer le déploiement
print("📤 Création du déploiement...")
deployment = ManagedOnlineDeployment(
    name="iris-classifier-deployment",
    endpoint_name="iris-classifier",
    model=f"{registered_model.name}:{registered_model.version}",
    environment=f"{registered_env.name}:{registered_env.version}",
    instance_type="Standard_F2s_v2",
    instance_count=1
)

ml_client.online_deployments.begin_create_or_update(deployment).result()
print("✅ Déploiement créé!")

endpoint = ml_client.online_endpoints.get("iris-classifier")
endpoint.traffic = {"iris-classifier-deployment": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

print(f"✅ Endpoint URL: {endpoint.scoring_uri}")
