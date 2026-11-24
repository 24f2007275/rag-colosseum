
import json
import os
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from huggingface_hub import InferenceClient

app = FastAPI(title="RAG Colosseum API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load knowledge base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "knowledge_base.json"), "r") as f:
    knowledge_base = json.load(f)

# Cloud embedding model
HF_TOKEN = os.environ["HF_TOKEN"]
MODEL_NAME = "BAAI/bge-small-en-v1.5"

client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN
)

# Normalize vectors
def normalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)

# Embed all passages using cloud API
passage_embeddings = normalize(
    client.feature_extraction(
        knowledge_base,
        model=MODEL_NAME
    )
)

print("Knowledge base:", len(knowledge_base))
print("Embedding shape:", passage_embeddings.shape)


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=3, ge=1, le=10)


MIN_SIMILARITY = 0.35


def retrieve(question, top_k):

    query_vector = normalize(
        client.feature_extraction(
            question,
            model=MODEL_NAME
        )
    )

    scores = np.dot(
        passage_embeddings,
        query_vector[0]
    )

    indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "index": int(i),
            "text": knowledge_base[i],
            "score": float(scores[i])
        }
        for i in indices
    ]


@app.get("/")
def home():
    return {
        "status": "online",
        "passages": len(knowledge_base)
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


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

    results = retrieve(question, request.top_k)

    best_score = results[0]["score"]

    if best_score < MIN_SIMILARITY:
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.2
        }

    answer = " ".join(
        item["text"] for item in results
    )

    return {
        "answer": answer,
        "sources": [
            item["index"] for item in results
        ],
        "scores": [
            round(max(0, min(1, item["score"])), 4)
            for item in results
        ],
        "confidence": round(
            min(0.99, max(0.61, best_score)), 4
        )
    }
