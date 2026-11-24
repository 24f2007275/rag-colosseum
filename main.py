
import json
import os
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# --------------------------------
# 1. Initialize FastAPI
# --------------------------------

app = FastAPI(title="RAG Colosseum API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------
# 2. Load Knowledge Base
# --------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(BASE_DIR, "knowledge_base.json")

with open(KB_PATH, "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

if not isinstance(knowledge_base, list) or not all(
    isinstance(p, str) for p in knowledge_base
):
    raise ValueError("Invalid knowledge_base.json format")

# --------------------------------
# 3. Load Embedding Model
# --------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

# Precompute passage embeddings
passage_embeddings = model.encode(
    knowledge_base,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=False
)

print("Knowledge base loaded:", len(knowledge_base))
print("Embeddings shape:", passage_embeddings.shape)

# --------------------------------
# 4. Request Schema
# --------------------------------

class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=3, ge=1, le=10)


# --------------------------------
# 5. Retrieval Function
# --------------------------------

MIN_SIMILARITY = 0.30


def retrieve_passages(question, top_k=3):

    query_embedding = model.encode(
        question,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # Calculate cosine similarity
    similarities = np.dot(
        passage_embeddings,
        query_embedding
    )

    # Get top relevant passages
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for idx in top_indices:
        score = float(similarities[idx])

        results.append({
            "index": int(idx),
            "text": knowledge_base[idx],
            "score": max(0.0, min(1.0, score))
        })

    return results


# --------------------------------
# 6. Answer Generation
# --------------------------------

def generate_answer(results):

    if not results:
        return None

    best_score = results[0]["score"]

    # Abstain if relevance is too low
    if best_score < MIN_SIMILARITY:
        return None

    # Combine relevant passages
    answer_parts = [
        item["text"] for item in results
    ]

    return " ".join(answer_parts)


# --------------------------------
# 7. Main Query Endpoint
# --------------------------------

@app.post("/query")
def query_endpoint(request: QueryRequest):

    question = request.question.strip()

    if not question:
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.1
        }

    # Retrieve relevant passages
    results = retrieve_passages(
        question,
        request.top_k
    )

    best_score = results[0]["score"] if results else 0.0

    # Check relevance
    if best_score < MIN_SIMILARITY:

        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.2
        }

    # Generate answer from retrieved evidence
    answer = generate_answer(results)

    sources = [
        item["index"] for item in results
    ]

    scores = [
        round(item["score"], 4) for item in results
    ]

    # Confidence estimate
    confidence = min(
        0.99,
        max(0.61, 0.61 + (best_score - MIN_SIMILARITY))
    )

    return {
        "answer": answer,
        "sources": sources,
        "scores": scores,
        "confidence": round(confidence, 4)
    }


# --------------------------------
# 8. Health Check
# --------------------------------

@app.get("/")
def home():

    return {
        "status": "online",
        "message": "RAG Colosseum API is running",
        "passages": len(knowledge_base)
    }
