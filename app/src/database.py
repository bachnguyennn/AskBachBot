from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential


COSMOS_ENDPOINT = "https://bach-rag-cosmos-central.documents.azure.com:443/"

credential = DefaultAzureCredential()

client = CosmosClient(
    COSMOS_ENDPOINT,
    credential=credential,
)

database = client.get_database_client("ask-bach")
container = database.get_container_client("content")