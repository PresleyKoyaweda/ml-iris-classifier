import os
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment
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

# Créer l'endpoint
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
deployment = ManagedOnlineDeployment(
    name="iris-classifier-deployment",
    endpoint_name="iris-classifier",
    model="iris-classifier:1",
    instance_type="Standard_F2s_v2",
    instance_count=1
)

print("📤 Création du déploiement...")
ml_client.online_deployments.begin_create_or_update(deployment).result()
print("✅ Déploiement créé!")

# Mettre à jour le trafic
endpoint = ml_client.online_endpoints.get("iris-classifier")
endpoint.traffic = {"iris-classifier-deployment": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

print(f"✅ Endpoint URL: {endpoint.scoring_uri}")
