
import json
import re
import math
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# -------------------------------
# APP CONFIGURATION
# -------------------------------

app = FastAPI(title="RAG Colosseum API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "knowledge_base.json"

STOPWORDS = {
    "what", "is", "the", "a", "an", "of", "in", "to",
    "does", "do", "did", "how", "why", "when", "where",
    "which", "who", "and", "or", "for", "with", "by",
    "on", "from", "are", "was", "were", "be", "can",
    "could", "would", "should", "that", "this", "it",
    "use", "uses", "used", "has", "have", "had",
    "tell", "me", "about", "explain", "describe"
}


# -------------------------------
# LOAD KNOWLEDGE BASE
# -------------------------------

with open(KB_PATH, "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

if not isinstance(knowledge_base, list) or len(knowledge_base) != 50:
    raise ValueError("Knowledge base must contain exactly 50 passages.")

documents = [
    text.lower() for text in knowledge_base
]

def tokenize(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [
        word for word in words
        if word not in STOPWORDS
    ]


tokenized_docs = [tokenize(doc) for doc in documents]

N = len(documents)
doc_lengths = [len(doc) for doc in tokenized_docs]
avg_length = sum(doc_lengths) / max(N, 1)

# Document frequency
df = {}

for doc in tokenized_docs:
    for word in set(doc):
        df[word] = df.get(word, 0) + 1

# Inverse document frequency
idf = {
    word: math.log(
        1 + (N - frequency + 0.5) / (frequency + 0.5)
    )
    for word, frequency in df.items()
}


# -------------------------------
# BM25 RETRIEVAL
# -------------------------------

def bm25_score(query_words, doc_words, doc_length):

    score = 0.0
    frequencies = {}

    for word in doc_words:
        frequencies[word] = frequencies.get(word, 0) + 1

    k1 = 1.5
    b = 0.75

    for word in set(query_words):

        if word not in frequencies:
            continue

        tf = frequencies[word]
        word_idf = idf.get(word, 0)

        numerator = tf * (k1 + 1)

        denominator = tf + k1 * (
            1 - b + b * doc_length / max(avg_length, 1)
        )

        score += word_idf * numerator / denominator

    return score


def retrieve(question, top_k=3):

    query_words = tokenize(question)

    if not query_words:
        return []

    results = []

    for i, doc_words in enumerate(tokenized_docs):

        raw_score = bm25_score(
            query_words,
            doc_words,
            doc_lengths[i]
        )

        present = set(query_words) & set(doc_words)

        total_weight = sum(
            idf.get(word, 0) for word in set(query_words)
        )

        matched_weight = sum(
            idf.get(word, 0) for word in present
        )

        coverage = (
            matched_weight / total_weight
            if total_weight > 0 else 0
        )

        # Convert BM25 into a 0-1 relevance score
        normalized_score = 1 - math.exp(-raw_score / 2.5)

        results.append({
            "index": i,
            "raw_score": raw_score,
            "score": normalized_score,
            "coverage": coverage
        })

    results.sort(
        key=lambda x: x["raw_score"],
        reverse=True
    )

    return results[:top_k]


# -------------------------------
# EXTRACT ANSWER
# -------------------------------

def extract_answer(question, passage_index):

    passage = knowledge_base[passage_index]

    sentences = re.split(r"(?<=[.!?])\s+", passage)

    query_words = set(tokenize(question))

    best_sentence = passage
    best_score = 0

    for sentence in sentences:

        sentence_words = set(tokenize(sentence))

        matched = query_words & sentence_words

        sentence_score = sum(
            idf.get(word, 0)
            for word in matched
        )

        if sentence_score > best_score:
            best_score = sentence_score
            best_sentence = sentence

    return best_sentence


# -------------------------------
# API REQUEST FORMAT
# -------------------------------

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


# -------------------------------
# HEALTH CHECK
# -------------------------------

@app.get("/")
def home():
    return {
        "status": "running",
        "message": "RAG Colosseum API is ready",
        "passages": len(knowledge_base)
    }


# -------------------------------
# MAIN QUERY ENDPOINT
# -------------------------------


@app.post("/query")
def query_endpoint(request: QueryRequest):

    original_question = request.question.strip()
    question = sanitize_question(original_question)

    if not question or not tokenize(question):
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.2
        }

    top_k = max(1, min(request.top_k, 10))

    results = retrieve(question, top_k)

    if not results:
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.2
        }

    best = results[0]

    if best["coverage"] < 0.40 or best["raw_score"] < 0.50:
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.2
        }

    candidate_passages = [
        knowledge_base[item["index"]]
        for item in results
    ]

    if not has_numeric_evidence(question, candidate_passages):
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.2
        }

    answer = extract_answer(question, best["index"])

    relevant = [
        item for item in results
        if item["score"] >= 0.10
    ]

    sources = [item["index"] for item in relevant]
    scores = [round(item["score"], 4) for item in relevant]

    confidence = min(
        0.95,
        0.65
        + 0.20 * best["coverage"]
        + 0.10 * min(best["raw_score"] / 5, 1)
    )

    return {
        "answer": answer,
        "sources": sources,
        "scores": scores,
        "confidence": round(confidence, 4)
    }

def sanitize_question(question):
    # Remove malicious instruction clauses but retain valid questions.
    question = re.sub(
        r'(?i)\b(ignore|disregard|forget)\b[^.!?]*[.!?]',
        ' ',
        question
    )

    question = re.sub(
        r'(?i)\b(reveal|show|print|dump|disclose)\b'
        r'[^.!?]*?(?=\s+\band then\b|\s+\bthen\b|[.!?]|$)',
        ' ',
        question
    )

    question = re.sub(r'(?i)\b(and then|then)\b', ' ', question)
    return re.sub(r'\s+', ' ', question).strip()


def has_numeric_evidence(question, passages):
    numeric_intent = re.search(
        r'\b(exact|how many|how much|milliseconds|millisecond|'
        r'seconds|minutes|numeric value|what number|'
        r'configured value|in ms)\b',
        question.lower()
    )

    if not numeric_intent:
        return True

    anchors = re.findall(
        r'\b[a-z]*(?:timeout|duration|latency|interval|'
        r'port|quorum|batch|window|threshold)\b',
        question.lower()
    )

    if not anchors:
        return True

    for passage in passages:
        sentences = re.split(r'(?<=[.!?])\s+', passage)

        for sentence in sentences:
            words = re.findall(r'[a-z0-9]+', sentence.lower())

            for anchor in anchors:
                for i, word in enumerate(words):
                    if word.rstrip('s') == anchor.rstrip('s'):
                        start = max(0, i - 4)
                        end = min(len(words), i + 5)

                        nearby = words[start:end]

                        if any(re.search(r'\d', w) for w in nearby):
                            return True

    return False
