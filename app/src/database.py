from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential

import os


COSMOS_ENDPOINT = "https://bach-rag-cosmos-central.documents.azure.com:443/"


credential = DefaultAzureCredential(
    managed_identity_client_id=os.getenv("AZURE_CLIENT_ID")
)

client = CosmosClient(
    COSMOS_ENDPOINT,
    credential=credential,
)

database = client.get_database_client("ask-bach")
container = database.get_container_client("content")