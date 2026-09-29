from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.graph import build_rag_graph

resources = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    resources["graph"] = build_rag_graph()
    yield
    resources.clear()


app = FastAPI(title="Agentic AI RAG API", lifespan=lifespan)
@app.get("/")
def root():
    return {"message": "Agentic AI RAG API. Open /docs to test POST /chat"}


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: List[str]
    confidence_score: float


@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    initial_state = {
        "question": request.query,
        "context": [],
        "pages": [],
        "retrieval_scores": [],
        "answer": "",
        "grounding_score": 0.0,
        "score": 0.0,
    }
    result = resources["graph"].invoke(initial_state)
    return QueryResponse(
        query=request.query,
        final_answer=result["answer"],
        retrieved_context_chunks=result["context"],
        confidence_score=result["score"],
    )


@app.get("/health")
def health():
    return {"status": "ok"}
