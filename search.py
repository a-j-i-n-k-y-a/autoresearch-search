# search.py
# This is the ONLY file the agent modifies.
# Start: basic BM25 baseline
# Goal: improve recall@K on benchmark queries

import numpy as np

def search(query, df, bm25, model, index, top_k=10):
    """
    Given a query and search resources, return top_k results.
    Each result is a dict with at least "title" and "overview".
    
    Resources available:
    - df: pandas DataFrame with columns title, overview, genres, text
    - bm25: BM25Okapi index
    - model: SentenceTransformer (all-MiniLM-L6-v2)
    - index: FAISS IndexFlatL2
    """
    tokenized = query.lower().split()
    scores    = bm25.get_scores(tokenized)
    indices   = np.argsort(scores)[::-1][:top_k]
    return df.iloc[indices][["title", "overview"]].to_dict("records")