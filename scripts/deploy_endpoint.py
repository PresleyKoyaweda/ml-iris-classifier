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

print("Création de l'endpoint...")
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
print("✅ Endpoint créé!")

print("\n📍 Endpoint URL: ", ml_client.online_endpoints.get("iris-classifier").scoring_uri)
