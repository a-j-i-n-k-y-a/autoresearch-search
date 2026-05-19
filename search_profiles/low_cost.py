import numpy as np
from rank_bm25 import BM25Okapi
import re

def _bm25_tokenize(text):
    return re.findall(r'\b\w\w+\b', text.lower())

def search(query, df, bm25, model, index, top_k=10):
    tokens = _bm25_tokenize(query)
    
    # Use pre-computed BM25 scores
    bm25_scores = bm25.get_scores(tokens)
    
    # Retrieve candidates using BM25 and Semantic search
    pool_size = 250
    idx_b = np.argpartition(bm25_scores, -pool_size)[-pool_size:]
    
    query_vec = model.encode([query]).astype("float32")
    dists, idx_s = index.search(query_vec, pool_size)
    
    # Merge candidates
    candidates = np.unique(np.concatenate([idx_b, idx_s[0]]))
    
    # Normalize scores
    s_b = bm25_scores[candidates]
    s_b = (s_b - s_b.min()) / (s_b.max() - s_b.min() + 1e-9)
    
    # Map vector distances
    dist_map = {idx: d for idx, d in zip(idx_s[0], dists[0])}
    s_s = np.array([1.0 / (1.0 + dist_map.get(i, 5.0)) for i in candidates])
    
    # Popularity boost
    pop = np.log1p(df['vote_count'].iloc[candidates].values)
    
    # Weighted fusion
    final_scores = (0.5 * s_b) + (0.5 * s_s) + (0.05 * pop)
    
    # Get top_k
    top_indices = candidates[np.argsort(final_scores)[-top_k:][::-1]]
    
    res = df.iloc[top_indices].copy()
    return res.to_dict("records")