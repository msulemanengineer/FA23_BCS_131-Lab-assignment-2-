# index_text_embeddings.py
import os
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import faiss
import json

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "neurips")
COLL_NAME = os.getenv("COLL_NAME", "papers")
EMB_MODEL = os.getenv("TEXT_EMB_MODEL", "all-MiniLM-L6-v2")
EMB_DIM = 384   # all-MiniLM-L6-v2 dim; adjust if you use other model
FAISS_INDEX_PATH = "text_faiss.index"
META_PATH = "text_meta.json"
EMB_SAVE_NPY = "text_embeddings.npy"

client = MongoClient(MONGO_URI)
coll = client[DB_NAME][COLL_NAME]
model = SentenceTransformer(EMB_MODEL)

def build_index():
    docs = list(coll.find({}, {"title":1, "authors":1}))
    texts = []
    ids = []
    for doc in docs:
        title = doc.get("title","")
        authors = ", ".join(doc.get("authors",[]))
        text = title + " | " + authors
        texts.append(text)
        ids.append(str(doc["_id"]))

    print("Encoding", len(texts), "documents...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    # save embeddings & metadata
    np.save(EMB_SAVE_NPY, embeddings)
    meta = {"ids": ids, "texts": texts}
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    # build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # dot product (we will normalize for cosine)
    # normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
    print("FAISS index built and saved:", FAISS_INDEX_PATH)

if __name__ == "__main__":
    build_index()
