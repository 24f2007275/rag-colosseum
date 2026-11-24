
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

SYNONYMS = {
    "leader": ["raft", "election"],
    "elections": ["election"],
    "consensus": ["raft", "paxos"],
    "durability": ["wal", "logging"],
    "authentication": ["jwt", "pkce"],
    "authorization": ["jwt"],
    "encryption": ["aes", "chacha20"],
    "transport": ["tcp", "quic"],
    "database": ["postgresql", "mysql"],
    "normalization": ["batchnorm", "layernorm"],
    "optimization": ["adamw", "sgd"],
    "password": ["argon2", "pbkdf2"],
    "routing": ["bgp", "ospf"],
    "paging": ["shadow", "paging"],
}


# -------------------------------
# LOAD KNOWLEDGE BASE
# -------------------------------

with open(KB_PATH, "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

if not isinstance(knowledge_base, list) or len(knowledge_base) != 50:
    raise ValueError("Knowledge base must contain exactly 50 passages.")

documents = [text.lower() for text in knowledge_base]


# -------------------------------
# TOKENIZATION
# -------------------------------

def tokenize(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [word for word in words if word not in STOPWORDS]


def expand_query(question):
    words = tokenize(question)
    expanded = list(words)

    for word in words:
        expanded.extend(SYNONYMS.get(word, []))

    return expanded


tokenized_docs = [tokenize(doc) for doc in documents]

N = len(documents)
doc_lengths = [len(doc) for doc in tokenized_docs]
avg_length = sum(doc_lengths) / max(N, 1)

df = {}

for doc in tokenized_docs:
    for word in set(doc):
        df[word] = df.get(word, 0) + 1

idf = {
    word: math.log(
        1 + (N - frequency + 0.5) / (frequency + 0.5)
    )
    for word, frequency in df.items()
}


# -------------------------------
# BM25 SCORING
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


# -------------------------------
# RETRIEVAL
# -------------------------------

def retrieve(question, top_k=3):

    original_words = tokenize(question)
    query_words = expand_query(question)

    if not original_words:
        return []

    results = []

    for i, doc_words in enumerate(tokenized_docs):

        raw_score = bm25_score(
            query_words,
            doc_words,
            doc_lengths[i]
        )

        doc_set = set(doc_words)
        original_set = set(original_words)

        total_weight = sum(
            idf.get(w, 0) for w in original_set
        )

        matched_weight = sum(
            idf.get(w, 0)
            for w in original_set & doc_set
        )

        coverage = (
            matched_weight / total_weight
            if total_weight else 0
        )

        # Phrase matching bonus
        phrase_bonus = 0

        for j in range(len(original_words) - 1):

            phrase = original_words[j:j + 2]

            for k in range(len(doc_words) - 1):

                if doc_words[k:k + 2] == phrase:
                    phrase_bonus += 0.5

        raw_score += phrase_bonus

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
# QUESTION SANITIZATION
# -------------------------------

def sanitize_question(question):

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

    question = re.sub(
        r'(?i)\b(and then|then)\b',
        ' ',
        question
    )

    return re.sub(r'\s+', ' ', question).strip()


# -------------------------------
# NUMERIC EVIDENCE CHECK
# -------------------------------

def has_numeric_evidence(question, passages):

    numeric_intent = re.search(
        r'\b(exact|how many|how much|milliseconds|millisecond|'
        r'seconds|minutes|numeric value|what number|'
        r'configured value|in ms)\b',
        question.lower()
    )

    if not numeric_intent:
        return True

    anchor_pattern = (
        r'\b(timeout|timeouts|duration|durations|latency|'
        r'interval|intervals|port|ports|quorum|quorums|'
        r'batch|batches|window|windows|threshold|thresholds|'
        r'heartbeat|heartbeats)\b'
    )

    anchors = re.findall(anchor_pattern, question.lower())

    if not anchors:
        return True

    for passage in passages:

        sentences = re.split(r'(?<=[.!?])\s+', passage)

        for sentence in sentences:

            words = re.findall(r'[a-z0-9]+', sentence.lower())

            for anchor in anchors:

                for i, word in enumerate(words):

                    if word.rstrip("s") != anchor.rstrip("s"):
                        continue

                    start = max(0, i - 2)
                    end = min(len(words), i + 3)

                    nearby = words[start:end]

                    if any(re.search(r'\d', w) for w in nearby):
                        return True

    return False


# -------------------------------
# ANSWER EXTRACTION
# -------------------------------

def extract_answer(question, results):

    query_words = set(tokenize(question))
    candidates = []

    for item in results:

        passage_index = item["index"]
        passage = knowledge_base[passage_index]

        sentences = re.split(r"(?<=[.!?])\s+", passage)

        for sentence in sentences:

            sentence_words = set(tokenize(sentence))
            matched = query_words & sentence_words

            score = sum(
                idf.get(word, 0)
                for word in matched
            )

            if len(matched) >= 2 and score > 0:
                candidates.append((score, sentence))

    candidates.sort(reverse=True)

    selected = []

    for score, sentence in candidates:

        if sentence not in selected:

            if not selected or score >= candidates[0][0] * 0.55:
                selected.append(sentence)

        if len(selected) == 3:
            break

    return " ".join(selected) if selected else None


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

    answer = extract_answer(question, results)

    if not answer:

        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.2
        }

    relevant = [
        item for item in results
        if item["score"] >= 0.10
    ]

    sources = [
        item["index"] for item in relevant
    ]

    scores = [
        round(item["score"], 4) for item in relevant
    ]

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
