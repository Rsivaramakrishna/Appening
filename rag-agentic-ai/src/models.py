from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from src import config


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=f"models/{config.EMBEDDING_MODEL}",
        output_dimensionality=config.EMBEDDING_DIMENSION,
        google_api_key=config.GOOGLE_API_KEY,
    )


def get_llm():
    return ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        temperature=0,
        google_api_key=config.GOOGLE_API_KEY,
    )


def to_text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "") if isinstance(part, dict) else str(part)
        for part in content
    )