"""Azure Cosmos DB access.

The client is created lazily on first use, so importing this module never opens a
network connection. That keeps unit tests and the offline eval free of Azure.
Auth is Microsoft Entra ID via DefaultAzureCredential: the user-assigned managed
identity in Azure (AZURE_CLIENT_ID), or your `az login` session locally. No keys.
"""

import os
from functools import lru_cache

COSMOS_ENDPOINT = os.getenv(
    "COSMOS_ENDPOINT", "https://bach-rag-cosmos-central.documents.azure.com:443/"
)
DATABASE_NAME = os.getenv("COSMOS_DATABASE", "ask-bach")


@lru_cache(maxsize=1)
def get_database():
    from azure.cosmos import CosmosClient
    from azure.identity import DefaultAzureCredential

    credential = DefaultAzureCredential(managed_identity_client_id=os.getenv("AZURE_CLIENT_ID"))
    return CosmosClient(COSMOS_ENDPOINT, credential=credential).get_database_client(DATABASE_NAME)


def get_container(name):
    """`content` holds whole documents and portfolio records; `chunks` holds embedded chunks."""
    return get_database().get_container_client(name)
