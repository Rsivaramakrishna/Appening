# RAG Chatbot: Agentic AI eBook

A Retrieval-Augmented Generation chatbot built with LangGraph, Pinecone, OpenAI and FastAPI. It answers only from the Agentic AI eBook and refuses everything else.

## Architecture

```
PDF -> PyPDFLoader -> RecursiveCharacterTextSplitter -> OpenAI embeddings -> Pinecone
                                                                              |
Question -> [retrieve] -> score check -> [generate] -> [grade] -> answer + chunks + confidence
                              |
                              +-> [refuse] -> refusal message
```

| Part | File | Job |
| --- | --- | --- |
| Config | `src/config.py` | Env vars and constants |
| Ingestion | `src/ingestion.py` | Download PDF, chunk, embed, upsert to Pinecone with page metadata |
| Graph | `src/graph.py` | LangGraph workflow: retrieve, generate, grade, refuse |
| API | `app.py` | FastAPI `/chat` endpoint |
| Tests | `tests_sample_queries.py` | 6 sample queries including an out-of-scope one |

### Graph nodes

- **retrieve**: top-4 chunks from Pinecone with cosine similarity scores.
- **route**: if the best score is below `MIN_RETRIEVAL_SCORE`, go to `refuse`. Otherwise go to `generate`.
- **generate**: `gpt-4o-mini` answers using only the retrieved context.
- **grade**: a second LLM call checks if the answer is supported by the context (groundedness, 0 to 1).
- **refuse**: returns a fixed refusal message with score 0.

### Confidence score

`0.7 * groundedness + 0.3 * average retrieval similarity`, rounded to 2 decimals. Refusals get 0.

### Chunking

800 characters, 100 overlap. Each chunk stores its page number and source file as metadata.

## Setup

```bash
git clone <your-repo-url>
cd rag-agentic-ai
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` with your OpenAI and Pinecone keys.

## Ingest the PDF

```bash
python -m src.ingestion
```

This downloads the PDF to `data/`, creates the Pinecone index if needed (dimension 1536, cosine), and uploads the chunks.

## Run the API

```bash
uvicorn app:app --reload
```

## Try it

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Agentic AI?"}'
```

Response:

```json
{
  "query": "What is Agentic AI?",
  "final_answer": "...",
  "retrieved_context_chunks": ["...", "..."],
  "confidence_score": 0.91
}
```

## Run the sample tests

With the API running in another terminal:

```bash
python tests_sample_queries.py
```

The last query ("What is the capital of France?") should be refused.
