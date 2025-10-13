# image_search.py
import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

IMG_MODEL = os.getenv("IMG_EMB_MODEL", "clip-ViT-B-32")  # try "clip-ViT-B-32" or "openai/clip-vit-base-patch32" depending on S-T version
EMB_DIR = "image_index"
FAISS_PATH = os.path.join(EMB_DIR, "images.faiss")
META_PATH = os.path.join(EMB_DIR, "image_meta.json")
EMB_NPY = os.path.join(EMB_DIR, "image_embs.npy")

os.makedirs(EMB_DIR, exist_ok=True)

model = SentenceTransformer(IMG_MODEL)  # this will load the image+text model; ensure it's available locally or online

def encode_image(image_path):
    img = Image.open(image_path).convert("RGB")
    emb = model.encode(img, convert_to_numpy=True, show_progress_bar=False)
    return emb

def index_images(folder):
    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        # glob manually to preserve order
        import glob
        files.extend(glob.glob(os.path.join(folder, ext)))
    files = sorted(files)
    embs = []
    for f in tqdm(files, desc="Encoding images"):
        try:
            e = encode_image(f)
            embs.append(e)
        except Exception as ex:
            print("Failed encode:", f, ex)
    embs = np.array(embs)
    # normalize
    faiss.normalize_L2(embs)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    faiss.write_index(index, FAISS_PATH)
    np.save(EMB_NPY, embs)
    meta = {"files": files, "dim": dim}
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    print("Indexed", len(files), "images. saved to", EMB_DIR)

def query_image(query_path, k=5):
    # load index
    index = faiss.read_index(FAISS_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    q_emb = encode_image(query_path)
    faiss.normalize_L2(q_emb.reshape(1, -1))
    D, I = index.search(q_emb.reshape(1, -1), k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        results.append({"file": meta["files"][idx], "score": float(score)})
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_folder", help="Folder with images to index")
    parser.add_argument("--query", help="Query image path")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    if args.index_folder:
        index_images(args.index_folder)
    if args.query:
        res = query_image(args.query, k=args.k)
        print("Top matches:")
        for r in res:
            print(r["file"], r["score"])
