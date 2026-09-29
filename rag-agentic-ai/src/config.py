import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")


PDF_URL = "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"
PDF_PATH = "data/Ebook-Agentic-AI.pdf"

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 768
LLM_MODEL = "gemini-3.8-flash"

EMBED_BATCH_SIZE = 90
EMBED_PAUSE_SECONDS = 65

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4

MIN_RETRIEVAL_SCORE = 0.30
REFUSAL_MESSAGE = "I cannot answer based on the provided document."

PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
