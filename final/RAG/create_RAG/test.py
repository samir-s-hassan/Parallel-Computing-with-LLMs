import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

# Load the model
generator = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to Qdrant
client = QdrantClient(
    url=os.getenv("QDRANT_URL"), 
    api_key=os.getenv("QDRANT_API_KEY"),
)

# The query to be run on the RAG
query_text = """"""

# Encode the query
query_embedding = generator.encode(query_text).tolist()

# Get the most similar vector
search_results = client.query_points(
    collection_name="updatedTextbooks",
    query=query_embedding,
    limit=1,
    with_payload=True
)
print(search_results.points[0].payload['text'])
