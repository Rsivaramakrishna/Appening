from typing import List, TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from src.models import get_embeddings, get_llm, to_text
from langchain_pinecone import PineconeVectorStore
from src import config


class AgentState(TypedDict):
    question: str
    context: List[str]
    pages: List[int]
    retrieval_scores: List[float]
    answer: str
    grounding_score: float
    score: float


class GroundingGrade(BaseModel):
    score: float = Field(description="Number from 0 to 1. 1 means every claim in the answer is supported by the context.")


def build_rag_graph():
    embeddings = get_embeddings()
    vectorstore = PineconeVectorStore(index_name=config.PINECONE_INDEX_NAME, embedding=embeddings)
    llm = get_llm()
    grader = llm.with_structured_output(GroundingGrade)

    def retrieve_node(state: AgentState):
        results = vectorstore.similarity_search_with_score(state["question"], k=config.TOP_K)
        return {
            "context": [doc.page_content for doc, _ in results],
            "pages": [int(doc.metadata.get("page", 0)) for doc, _ in results],
            "retrieval_scores": [float(score) for _, score in results],
        }

    def route_after_retrieve(state: AgentState):
        scores = state["retrieval_scores"]
        if not scores or max(scores) < config.MIN_RETRIEVAL_SCORE:
            return "refuse"
        return "generate"

    def refuse_node(state: AgentState):
        return {"answer": config.REFUSAL_MESSAGE, "grounding_score": 0.0, "score": 0.0}

    def generate_node(state: AgentState):
        context_str = "\n\n".join(
            f"[Page {page}] {text}" for text, page in zip(state["context"], state["pages"])
        )
        prompt = (
            "You are a strict assistant. Answer the question using ONLY the context below. "
            "Do not use outside knowledge. "
            f"If the context does not contain the answer, reply exactly: '{config.REFUSAL_MESSAGE}'\n\n"
            f"Context:\n{context_str}\n\nQuestion: {state['question']}"
        )
        response = llm.invoke(prompt)
        return {"answer": to_text(response.content).strip()}

    def grade_node(state: AgentState):
        if config.REFUSAL_MESSAGE in state["answer"]:
            return {"grounding_score": 0.0, "score": 0.0}
        context_str = "\n\n".join(state["context"])
        prompt = (
            "Check if the answer is fully supported by the context. "
            "Give a score from 0 to 1.\n\n"
            f"Context:\n{context_str}\n\nAnswer:\n{state['answer']}"
        )
        grade = grader.invoke(prompt)
        grounding = max(0.0, min(1.0, grade.score))
        scores = state["retrieval_scores"]
        retrieval = sum(scores) / len(scores) if scores else 0.0
        retrieval = max(0.0, min(1.0, retrieval))
        final = 0.7 * grounding + 0.3 * retrieval
        return {"grounding_score": grounding, "score": round(final, 2)}

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("refuse", refuse_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_conditional_edges(
        "retrieve", route_after_retrieve, {"generate": "generate", "refuse": "refuse"}
    )
    workflow.add_edge("generate", "grade")
    workflow.add_edge("grade", END)
    workflow.add_edge("refuse", END)

    return workflow.compile()
