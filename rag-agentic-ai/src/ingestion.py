import os
import time
import requests
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.models import get_embeddings
from langchain_pinecone import PineconeVectorStore
from src import config


def download_pdf(url: str, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return path
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with open(path, "wb") as f:
        f.write(response.content)
    return path


def ensure_index(pc: Pinecone, name: str) -> None:
    if name not in pc.list_indexes().names():
        pc.create_index(
            name=name,
            dimension=config.EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )
    while not pc.describe_index(name).status["ready"]:
        time.sleep(1)


def load_and_split(pdf_path: str):
    docs = PyPDFLoader(pdf_path).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        chunk.metadata["page"] = int(chunk.metadata.get("page", 0)) + 1
        chunk.metadata["source"] = "Ebook-Agentic-AI.pdf"
    return chunks


def run_ingestion() -> PineconeVectorStore:
    pdf_path = download_pdf(config.PDF_URL, config.PDF_PATH)
    chunks = load_and_split(pdf_path)

    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    ensure_index(pc, config.PINECONE_INDEX_NAME)

    store = PineconeVectorStore(
        index_name=config.PINECONE_INDEX_NAME,
        embedding=get_embeddings(),
    )

    size = config.EMBED_BATCH_SIZE
    for start in range(0, len(chunks), size):
        batch = chunks[start:start + size]
        ids = [f"chunk-{start + i}" for i in range(len(batch))]
        store.add_documents(batch, ids=ids)
        print(f"Indexed {min(start + size, len(chunks))}/{len(chunks)} chunks")
        if start + size < len(chunks):
            time.sleep(config.EMBED_PAUSE_SECONDS)

    print(f"Done. Index '{config.PINECONE_INDEX_NAME}' is ready")
    return store


if __name__ == "__main__":
    run_ingestion()
