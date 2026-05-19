import numpy as np
from rank_bm25 import BM25Okapi
import re

def _bm25_tokenize(text):
    return re.findall(r'\b\w\w+\b', text.lower())

def search(query, df, bm25, model, index, top_k=10):
    tokens = _bm25_tokenize(query)
    bm25_scores = bm25.get_scores(tokens)
    
    # Increase candidate pool slightly to capture better recall
    pool_size = 300
    idx_b = np.argpartition(bm25_scores, -pool_size)[-pool_size:]
    
    query_vec = model.encode([query]).astype("float32")
    dists, idx_s = index.search(query_vec, pool_size)
    
    # Consolidate candidates
    candidates = np.unique(np.concatenate([idx_b, idx_s[0]]))
    
    # Vectorized score normalization
    s_b = bm25_scores[candidates]
    s_b = (s_b - s_b.min()) / (s_b.max() - s_b.min() + 1e-9)
    
    dist_map = {idx: d for idx, d in zip(idx_s[0], dists[0])}
    s_s = np.array([1.0 / (1.0 + dist_map.get(i, 5.0)) for i in candidates])
    
    # Genre boost based on query keywords
    genre_col = df['genres'].iloc[candidates].values
    genre_boost = np.array([0.15 if any(t in str(g).lower() for t in tokens) else 0.0 for g in genre_col])
    
    # Popularity bias
    pop = np.log1p(df['vote_count'].iloc[candidates].values)
    pop_norm = (pop - pop.mean()) / (pop.std() + 1e-9)
    
    # Weighted fusion
    final_scores = (0.4 * s_b) + (0.5 * s_s) + (0.1 * pop_norm) + genre_boost
    
    # Sort and return
    top_indices = candidates[np.argsort(final_scores)[-top_k:][::-1]]
    
    res = df.iloc[top_indices].copy()
    res['score'] = final_scores[np.argsort(final_scores)[-top_k:][::-1]]
    return res.to_dict("records")