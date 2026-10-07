from pathlib import Path
from src.database import container


DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "documents"


for path in sorted(DOCS_DIR.glob("*.txt")):
    text = path.read_text(encoding="utf-8").strip()
    
    if not text:
        continue
    
    item = {
        "id": path.stem,
        "type": "document",
        "source" :path.name,
        "text" :text,
    }

    container.upsert_item(item)
    print(f"Uploaded {path.name} ")



items = list(
    container.query_items(
        query="SELECT c.id, c.source FROM c WHERE c.type = 'document'",
        enable_cross_partition_query=True,
    )
)

for item in items:
    print(item)