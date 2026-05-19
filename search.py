import numpy as np
from rank_bm25 import BM25Okapi

def _bm25_tokenize(text):
    import re
    return re.findall(r'\b\w\w+\b', text.lower())

def search(query, df, bm25, model, index, top_k=10):
    tokens = _bm25_tokenize(query)
    
    # Efficiently retrieve candidates (pool size 250 for balance)
    bm25_scores = bm25.get_scores(tokens)
    idx_b = np.argpartition(bm25_scores, -250)[-250:]
    
    query_vec = model.encode([query]).astype("float32")
    dists, idx_s = index.search(query_vec, 250)
    
    # Merge candidates
    candidates = np.unique(np.concatenate([idx_b, idx_s[0]]))
    
    # Normalize scores locally
    s_b = bm25_scores[candidates]
    s_b = (s_b - s_b.min()) / (s_b.max() - s_b.min() + 1e-9)
    
    # Map vector distances
    dist_map = {idx: d for idx, d in zip(idx_s[0], dists[0])}
    s_s = np.array([dist_map.get(i, 5.0) for i in candidates])
    s_s = 1.0 / (1.0 + s_s)
    
    # Metadata signals
    pop = np.log1p(df['vote_count'].iloc[candidates].values)
    pop = (pop - pop.mean()) / (pop.std() + 1e-9)
    
    # Simple weighted fusion
    final_scores = (0.4 * s_b) + (0.5 * s_s) + (0.1 * pop)
    
    # Direct sort and slice
    top_indices = candidates[np.argsort(final_scores)[-top_k:][::-1]]
    
    res = df.iloc[top_indices].copy()
    res['score'] = final_scores[np.argsort(final_scores)[-top_k:][::-1]]
    return res.to_dict("records")