import os
import json
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from dotenv import load_dotenv

load_dotenv()

def upload_to_qdrant():
    embeddings_dir="./embeddings"
    collection_name="updatedTextbooks"
    
    # Initialize client
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"), 
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    # Check/create collection
    if not client.collection_exists(collection_name):
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

    point_id = 0
    for file_name in os.listdir(embeddings_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(embeddings_dir, file_name)

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            chunks = data.get("chunks", [])
            embeddings = data.get("embeddings", [])

            points = [
                PointStruct(
                    id=point_id + i,
                    vector=embedding,
                    payload={"text": chunks[i], "source_file": file_name}
                )
                for i, embedding in enumerate(embeddings)
            ]

            client.upsert(collection_name=collection_name, points=points)
            point_id += len(embeddings)

            print(f"Uploaded {len(embeddings)} vectors from {file_name}")

if __name__ == "__main__":
    upload_to_qdrant()



