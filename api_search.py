# api_search.py
import os
import json
from fastapi import FastAPI, Query
from pydantic import BaseModel
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from bson import ObjectId

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "neurips")
COLL_NAME = os.getenv("COLL_NAME", "papers")
FAISS_INDEX_PATH = "text_faiss.index"
META_PATH = "text_meta.json"
TEXT_EMB_MODEL = os.getenv("TEXT_EMB_MODEL", "all-MiniLM-L6-v2")

client = MongoClient(MONGO_URI)
coll = client[DB_NAME][COLL_NAME]

# load FAISS & meta
index = faiss.read_index(FAISS_INDEX_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)
meta_ids = meta["ids"]  # corresponds to positions in index

model = SentenceTransformer(TEXT_EMB_MODEL)

app = FastAPI()

def mongo_text_search(q, limit=10):
    # uses MongoDB text search; needs text index on title & authors
    res_cursor = coll.find({"$text": {"$search": q}}, {"score": {"$meta": "textScore"}, "title":1, "authors":1, "link":1}).sort([("score", {"$meta":"textScore"})]).limit(limit)
    results = []
    for d in res_cursor:
        results.append({"_id": str(d["_id"]), "title": d.get("title"), "authors": d.get("authors"), "link": d.get("link"), "text_score": d.get("score",0)})
    return results

@app.get("/search/keyword")
def search_keyword(q: str = Query(..., min_length=1), limit: int = 10):
    results = mongo_text_search(q, limit)
    return {"query": q, "results": results}

@app.get("/search/hybrid")
def search_hybrid(q: str = Query(..., min_length=1), limit: int = 10, alpha: float = 0.5):
    # alpha is weight of embedding score (0..1), (1-alpha) weight for text score
    q_emb = model.encode([q], convert_to_numpy=True)
    # normalize
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k=limit)
    neighbors = []
    # D are inner products (cosine after normalization)
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        doc_id = meta_ids[idx]
        doc = coll.find_one({"_id": ObjectId(doc_id)})
        text_score = 0.0
        # try text score via $text search on the single document
        # cheaper: we can compute a simple BM25 via Mongo text search across small set, but for simplicity we'll call mongo_text_search and find score if exists
        # Here: do a text search with filter on _id
        # BUT Mongo doesn't return score unless using $text against the collection, so we perform a limited $text query:
        ts = list(coll.find({"$text":{"$search": q}, "_id": ObjectId(doc_id)}, {"score": {"$meta":"textScore"}}))
        if ts:
            text_score = ts[0].get("score", 0.0)
        # normalize text_score later. For now, collect
        neighbors.append({
            "mongo_id": str(doc["_id"]),
            "title": doc.get("title"),
            "authors": doc.get("authors"),
            "link": doc.get("link"),
            "emb_score": float(score),
            "text_score": float(text_score)
        })

    # Normalize embedding scores to [0,1]; they are cosine similarities in [-1,1]
    emb_scores = np.array([n["emb_score"] for n in neighbors])
    if len(emb_scores)>0:
        # shift to [0,1]
        emb_norm = (emb_scores - emb_scores.min()) / (emb_scores.max() - emb_scores.min() + 1e-12)
    else:
        emb_norm = emb_scores

    # normalize text_score
    text_scores = np.array([n["text_score"] for n in neighbors])
    if text_scores.size:
        text_norm = (text_scores - text_scores.min()) / (text_scores.max() - text_scores.min() + 1e-12)
    else:
        text_norm = text_scores

    combined = []
    for i, n in enumerate(neighbors):
        e = float(emb_norm[i]) if len(emb_norm)>0 else 0.0
        t = float(text_norm[i]) if len(text_norm)>0 else 0.0
        combined_score = alpha * e + (1-alpha) * t
        n["emb_score_norm"] = e
        n["text_score_norm"] = t
        n["combined_score"] = combined_score
        combined.append(n)
    combined = sorted(combined, key=lambda x: x["combined_score"], reverse=True)
    return {"query": q, "alpha": alpha, "results": combined[:limit]}

# Run with:
# uvicorn api_search:app --reload --port 8000
