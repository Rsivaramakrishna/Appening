# RAG Chatbot: Agentic AI eBook

A Retrieval-Augmented Generation (RAG) chatbot that answers questions only from the **Agentic AI eBook**. It is built with Python, LangGraph, Pinecone, Google Gemini and FastAPI.

If the answer is not in the eBook, the bot refuses instead of guessing.

## Features

- PDF ingestion: download, extract, chunk, embed, and store in Pinecone with page numbers
- LangGraph workflow with four nodes: retrieve, generate, grade, refuse
- Strict grounding: the LLM may use only the retrieved context
- Groundedness check: a second LLM call scores how well the answer is supported
- Refusal for out-of-scope questions, both by a similarity threshold and by the prompt
- FastAPI endpoint that returns the answer, the retrieved chunks and a confidence score

## Tech Stack

| Part | Technology |
| --- | --- |
| Language | Python 3.10+ |
| Orchestration | LangGraph |
| Vector database | Pinecone (serverless, cosine, 768 dimensions) |
| Embeddings | Google `gemini-embedding-001` |
| LLM | Google Gemini (`gemini-3.8-flash`) |
| PDF and chunking | PyPDF, LangChain `RecursiveCharacterTextSplitter` |
| API | FastAPI + Uvicorn |

## Architecture

```
INGESTION (run once)
PDF -> PyPDFLoader -> RecursiveCharacterTextSplitter -> Gemini embeddings -> Pinecone
                      (800 chars, 100 overlap)          (768 dim)            (text + page metadata)

QUERY (per request)
Question -> [retrieve] -> best score >= threshold? -- yes --> [generate] -> [grade] -> END
                |                                   |
                |                                   no
                |                                   v
                +-----------------------------> [refuse] -> END
```

### Graph nodes

| Node | What it does |
| --- | --- |
| `retrieve` | Gets the top 4 chunks from Pinecone with cosine similarity scores and page numbers |
| route | If the best score is below `MIN_RETRIEVAL_SCORE`, go to `refuse`. Otherwise go to `generate` |
| `generate` | Gemini answers using only the retrieved context. If the context has no answer, it returns a fixed refusal message |
| `grade` | A second Gemini call scores from 0 to 1 how well the answer is supported by the context |
| `refuse` | Returns the refusal message with confidence 0 |

### Confidence score

```
confidence = 0.7 * groundedness + 0.3 * average retrieval similarity
```

The value is rounded to 2 decimals. Refusals always get 0.0.

## Project Structure

```
rag-agentic-ai/
├── data/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── config.py        Environment variables and constants
│   ├── models.py        Gemini embeddings and LLM setup
│   ├── ingestion.py     PDF download, chunking, embedding, Pinecone upload
│   └── graph.py         LangGraph workflow and state
├── app.py               FastAPI application
├── tests_sample_queries.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <your-repo-url>
cd rag-agentic-ai
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your API keys

```bash
cp .env.example .env
```

Edit `.env`:

```
GOOGLE_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-ai-index
```

Keys: Gemini from Google AI Studio, Pinecone from the Pinecone console (free tier works).

### 3. Ingest the eBook

```bash
python -m src.ingestion
```

This will:
1. Download the PDF to `data/Ebook-Agentic-AI.pdf`
2. Split it into chunks of 800 characters with 100 overlap
3. Create the Pinecone index if it does not exist (768 dimensions, cosine)
4. Embed and upload the chunks in batches

The Gemini free tier allows 100 embedding requests per minute, so the script uploads 90 chunks at a time and waits about 65 seconds between batches. This is set in `src/config.py`.

If an index with the same name already exists with a different dimension, delete it in the Pinecone console and run the command again.

### 4. Start the API

```bash
uvicorn app:app --reload
```

Open http://127.0.0.1:8000/docs to try the endpoint in the browser.

### 5. Ask a question

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Agentic AI?"}'
```

## API

### `POST /chat`

Request:

```json
{
  "query": "What is Agentic AI?"
}
```

Response:

```json
{
  "query": "What is Agentic AI?",
  "final_answer": "...",
  "retrieved_context_chunks": ["chunk 1 text", "chunk 2 text"],
  "confidence_score": 0.91
}
```

### `GET /health`

Returns `{"status": "ok"}`.

## Testing

Start the API in one terminal. In a second terminal run:

```bash
python tests_sample_queries.py
```

The script sends six queries:

1. Core definition of Agentic AI
2. Main architectural components of agentic systems
3. Real-world industry use cases
4. Agentic AI vs traditional generative AI chatbots
5. Key challenges and limitations
6. "What is the capital of France?" (out of scope, must be refused)

The script waits between queries to stay inside the free-tier rate limit.

### Sample output

> Replace this section with your real output from `tests_sample_queries.py`.

```
QUERY: What is the core definition of Agentic AI as outlined in the eBook?
ANSWER: ...
CONFIDENCE: ...

QUERY: What is the capital of France?
ANSWER: I cannot answer based on the provided document.
CONFIDENCE: 0.0
```

## Configuration

All settings are in `src/config.py`.

| Setting | Value | Meaning |
| --- | --- | --- |
| `CHUNK_SIZE` | 800 | Characters per chunk |
| `CHUNK_OVERLAP` | 100 | Overlap between chunks |
| `TOP_K` | 4 | Chunks retrieved per question |
| `MIN_RETRIEVAL_SCORE` | 0.30 | Below this best score, the bot refuses without calling the LLM |
| `EMBEDDING_MODEL` | gemini-embedding-001 | Embedding model |
| `EMBEDDING_DIMENSION` | 768 | Must match the Pinecone index |
| `LLM_MODEL` | gemini-3.8-flash | Answer and grader model |

## Design Decisions

- **Two layers of refusal.** The similarity threshold stops clearly unrelated questions early and saves an LLM call. The strict prompt handles the cases the threshold lets through.
- **A real confidence score.** The score combines an LLM groundedness check with the retrieval similarity, instead of a fixed number.
- **Page numbers in metadata.** Each chunk stores its page, so answers can be traced back to the eBook.
- **Batched ingestion with fixed IDs.** Batches respect the free-tier limit, and fixed chunk IDs mean running ingestion again does not create duplicates.
- **Models in one place.** `src/models.py` builds the embeddings and LLM, so switching providers means changing one file.

## Limitations

- The groundedness score comes from an LLM, so it is an estimate and not a guarantee.
- The refusal threshold was tuned by hand on this one eBook. Another document may need a different value.
- The Gemini free tier limits requests per minute, so answers can be slow or return a 429 error under heavy use.
- No conversation memory. Each question is answered on its own.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `401 Invalid API key` from Pinecone | Check `.env` is saved, has no quotes or spaces, and uses the real key |
| `Vector dimension does not match the index` | Delete the old index in Pinecone and run ingestion again |
| `429 RESOURCE_EXHAUSTED` | Free-tier rate limit. Wait one minute and retry |
| `404 model not found` | Set a model your key can use in `src/config.py` |
| `Connection refused` in tests | The API is not running. Start `uvicorn` first |
