import time
import requests

API_URL = "http://localhost:8000/chat"
PAUSE_SECONDS = 20

QUERIES = [
    "What is the core definition of Agentic AI as outlined in the eBook?",
    "What are the main architectural components required to build agentic systems?",
    "What real-world industry use cases for Agentic AI are discussed in the eBook?",
    "How does Agentic AI differ from traditional generative AI chatbots according to the text?",
    "What key challenges or limitations of Agentic AI are mentioned in the document?",
    "What is the capital of France?",
]


def ask(query):
    for attempt in range(3):
        response = requests.post(API_URL, json={"query": query}, timeout=180)
        if response.status_code == 200:
            return response.json()
        print(f"Attempt {attempt + 1} failed with {response.status_code}, waiting 60s")
        time.sleep(60)
    return None


def main():
    for index, query in enumerate(QUERIES):
        data = ask(query)
        print("=" * 80)
        print("QUERY:", query)
        if data is None:
            print("FAILED after 3 attempts")
        else:
            print("ANSWER:", data["final_answer"])
            print("CONFIDENCE:", data["confidence_score"])
            print("CHUNKS RETRIEVED:", len(data["retrieved_context_chunks"]))
            chunks = data["retrieved_context_chunks"]
            print("FIRST CHUNK:", chunks[0][:200] if chunks else "none")
        if index < len(QUERIES) - 1:
            time.sleep(PAUSE_SECONDS)
    print("=" * 80)


if __name__ == "__main__":
    main()