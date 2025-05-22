import os
import json
from sentence_transformers import SentenceTransformer
import tiktoken
from markdown import markdown
from bs4 import BeautifulSoup

class EmbeddingGenerator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def chunk_text_by_tokens(self, text, chunk_size, encoding_name="cl100k_base"):
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)
        return [encoding.decode(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]

    def generate_embeddings(self, chunks):
        return self.model.encode(chunks, convert_to_numpy=True).tolist()

    def process_text(self, text, chunk_size=1000):
        chunks = self.chunk_text_by_tokens(text, chunk_size)
        embeddings = self.generate_embeddings(chunks)
        return chunks, embeddings


def markdown_to_text(md_content):
    html = markdown(md_content)
    soup = BeautifulSoup(html, features="html.parser")
    return soup.get_text(separator=" ")


def process_directory():
    input_dir="./output"
    output_dir="./embeddings"
    
    print(f"Processing files in {input_dir}...")

    os.makedirs(output_dir, exist_ok=True)
    generator = EmbeddingGenerator()
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".md"):
            input_path = os.path.join(input_dir, filename)
            with open(input_path, "r", encoding="utf-8") as file:
                text = file.read()
                text = markdown_to_text(text)
                text = text.replace("\n", " ")

            chunks, embeddings = generator.process_text(text, 800)

            output_data = {
                "file": filename,
                "chunks": chunks,
                "embeddings": embeddings
            }

            output_filename = os.path.splitext(filename)[0] + ".json"
            output_path = os.path.join(output_dir, output_filename)
            with open(output_path, "w", encoding="utf-8") as out_file:
                json.dump(output_data, out_file, ensure_ascii=False, indent=2)

            print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    process_directory()
