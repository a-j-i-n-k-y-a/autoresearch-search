import numpy as np
from rank_bm25 import BM25Okapi

def _bm25_tokenize(text):
    import re
    return re.findall(r'\b\w\w+\b', text.lower())

def search(query, df, bm25, model, index, top_k=10):
    tokens = _bm25_tokenize(query)
    
    # BM25 Retrieval
    bm25_scores = bm25.get_scores(tokens)
    top_bm25_idx = np.argpartition(bm25_scores, -200)[-200:]
    
    # Semantic Retrieval
    query_vec = model.encode([query]).astype("float32")
    dists, idxs = index.search(query_vec, 200)
    
    # Combine unique candidates
    candidate_indices = np.unique(np.concatenate([top_bm25_idx, idxs[0]]))
    
    # Scoring features
    c_bm25 = bm25_scores[candidate_indices]
    c_bm25 = (c_bm25 - c_bm25.min()) / (c_bm25.max() - c_bm25.min() + 1e-9)
    
    idx_map = {idx: 1.0/(1.0+d) for idx, d in zip(idxs[0], dists[0])}
    c_semantic = np.array([idx_map.get(i, 0.0) for i in candidate_indices])
    
    # Genre signal: binary match
    c_genres = df.iloc[candidate_indices]['genres'].str.lower()
    genre_match = np.array([1.0 if any(g in q_part for g in tokens) else 0.5 for q_part, genres in zip([query]*len(c_genres), c_genres)])
    
    # Popularity bias: log density
    pop = np.log1p(df.iloc[candidate_indices]['vote_count'])
    pop = (pop - pop.mean()) / (pop.std() + 1e-9)
    
    # Weighted ensemble
    final_scores = (0.35 * c_bm25) + (0.45 * c_semantic) + (0.1 * genre_match) + (0.1 * pop)
    
    # Sort and return
    res = df.iloc[candidate_indices].copy()
    res['score'] = final_scores
    return res.sort_values("score", ascending=False).head(top_k).to_dict("records")