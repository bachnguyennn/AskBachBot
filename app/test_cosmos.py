from src.database import container


item = {
    "id": "test-item",
    "type": "test",
    "message": "Hello from Python"
}

container.upsert_item(item)

result = container.read_item(
    item="test-item",
    partition_key="test"
)

container.delete_item(
    item="test-item",
    partition_key="test"
)
print(result)