# prepare.py
# Fixed constants, data prep, and evaluation
# DO NOT MODIFY — this is the ground truth

import numpy as np
import pandas as pd
import time
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle

# ─── FIXED CONSTANTS ───────────────────────────────────────
DATASET_SIZE    = 100000      # number of movies to use
TOP_K           = 10          # recall measured at top 10
CACHE_DIR       = "data/"
EXPERIMENT_TIME = 60          # max seconds per experiment

# ─── BENCHMARK QUERIES (ground truth, never change) ────────
BENCHMARK_QUERIES = [
    {
        "query": "dream heist movie Leonardo DiCaprio layers of subconscious",
        "expected": ["Inception"]
    },
    {
        "query": "astronaut stranded in space wormhole black hole",
        "expected": ["Interstellar", "Gravity"]
    },
    {
        "query": "robot humanoid artificial intelligence consciousness",
        "expected": ["Ex Machina", "A.I. Artificial Intelligence"]
    },
    {
        "query": "psychological thriller unreliable narrator mind bending twist",
        "expected": ["Shutter Island", "Black Swan"]
    },
    {
        "query": "time machine going back to the future paradox",
        "expected": ["Back to the Future", "Looper"]
    },
]

# ─── DATA PREP (run once) ──────────────────────────────────
def prepare_data():
    os.makedirs(CACHE_DIR, exist_ok=True)

    if not os.path.exists(f"{CACHE_DIR}movies.pkl"):
        print("Downloading dataset...")
        dataset = load_dataset("wykonos/movies", split="train")
        df = pd.DataFrame(dataset)
        df = df[["title", "overview", "genres", "vote_average", "vote_count"]].dropna()
        df = df[df["vote_count"] > 100]
        df = df[df["vote_average"] > 5.0]
        df = df.head(DATASET_SIZE).reset_index(drop=True)
        df["text"] = df["title"] + " " + df["overview"] + " " + df["genres"]
        df.to_pickle(f"{CACHE_DIR}movies.pkl")
        print(f"Saved {len(df)} movies")

    if not os.path.exists(f"{CACHE_DIR}faiss.index"):
        print("Building vector index...")
        df = pd.read_pickle(f"{CACHE_DIR}movies.pkl")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, f"{CACHE_DIR}faiss.index")
        with open(f"{CACHE_DIR}embeddings.pkl", "wb") as f:
            pickle.dump(embeddings, f)
        print("Index saved")

    print("Data ready.")

# ─── LOAD RESOURCES ───────────────────────────────────────
def load_resources():
    df    = pd.read_pickle(f"{CACHE_DIR}movies.pkl")
    texts = df["text"].tolist()

    bm25  = BM25Okapi([t.lower().split() for t in texts])
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(f"{CACHE_DIR}faiss.index")

    return df, bm25, model, index

# ─── EVALUATION (ground truth metric, never change) ────────
def evaluate(search_fn, df, bm25, model, index):
    """
    Run all benchmark queries through search_fn.
    Returns recall@K — the single metric to optimize.
    Higher is better.
    """
    total_recall = 0.0

    for item in BENCHMARK_QUERIES:
        query    = item["query"]
        expected = [e.lower() for e in item["expected"]]

        results = search_fn(query, df, bm25, model, index, top_k=TOP_K)
        retrieved = [r["title"].lower() for r in results]

        found = sum(1 for e in expected if e in retrieved)
        recall = found / len(expected)
        total_recall += recall

    avg_recall = total_recall / len(BENCHMARK_QUERIES)
    return round(avg_recall, 6)


if __name__ == "__main__":
    prepare_data()
    print("Setup complete. Run agent_loop.py to start experiments.")