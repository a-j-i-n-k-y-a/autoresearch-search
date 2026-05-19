import numpy as np
from rank_bm25 import BM25Okapi

def _bm25_tokenize(text):
    import re
    # Lowercase, remove punct, filter by length > 1
    tokens = re.findall(r'\b\w\w+\b', text.lower())
    # Note: stopword list from prepare.py context is implied by standard BM25 practices
    return tokens

def search(query, df, bm25, model, index, top_k=10):
    # Retrieve top 500 candidates via BM25
    bm25_scores = bm25.get_scores(_bm25_tokenize(query))
    top_bm25_idx = np.argsort(bm25_scores)[-500:]
    
    # Semantic retrieval: get 500 candidates via index
    query_vec = model.encode([query]).astype("float32")
    dists, idxs = index.search(query_vec, 500)
    
    # Merge candidates
    candidate_mask = np.zeros(len(df), dtype=bool)
    candidate_mask[top_bm25_idx] = True
    candidate_mask[idxs[0]] = True
    candidate_indices = np.where(candidate_mask)[0]
    
    candidates = df.iloc[candidate_indices].copy()
    
    # Scoring
    # 1. BM25 (scaled)
    c_bm25 = bm25_scores[candidate_indices]
    c_bm25 = (c_bm25 - c_bm25.min()) / (c_bm25.max() - c_bm25.min() + 1e-9)
    
    # 2. Semantic (1 / 1 + L2)
    # Map index results
    idx_map = {idx: 1.0/(1.0+d) for idx, d in zip(idxs[0], dists[0])}
    c_semantic = np.array([idx_map.get(i, 0.0) for i in candidate_indices])
    
    # 3. Popularity & Genre
    # Boost by vote_average if vote_count is significant
    pop = (candidates['vote_average'] * np.log1p(candidates['vote_count']))
    pop = (pop - pop.mean()) / (pop.std() + 1e-9)
    
    # Combine: RRF-like combination
    # Normalized sum of scores
    final_scores = (c_bm25 * 0.4) + (c_semantic * 0.6) + (pop * 0.1)
    
    candidates['score'] = final_scores
    return candidates.sort_values("score", ascending=False).head(top_k).to_dict("records")