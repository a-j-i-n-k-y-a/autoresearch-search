import numpy as np
from rank_bm25 import BM25Okapi
import faiss

def search(query, df, bm25, model, index, top_k=10):
    tokenized = query.lower().split()
    query_vec = model.encode([query]).astype("float32")
    
    vec_distances, vec_ids = index.search(query_vec, top_k * 10)
    bm25_scores = bm25.get_scores(tokenized)
    
    # Simple hybrid: top BM25 + top Vector
    bm25_cands = np.argsort(bm25_scores)[-100:]
    vec_cands = vec_ids[0][vec_ids[0] != -1]
    cand_ids = np.unique(np.concatenate([bm25_cands, vec_cands]))
    
    # Normalize scores
    b_scores = bm25_scores[cand_ids]
    b_scores = (b_scores - b_scores.min()) / (b_scores.max() - b_scores.min() + 1e-9)
    
    v_scores = np.full(len(cand_ids), 100.0)
    for i, cid in enumerate(cand_ids):
        match = np.where(vec_ids[0] == cid)[0]
        if len(match) > 0:
            v_scores[i] = vec_distances[0][match[0]]
    v_scores = 1.0 - (v_scores / (v_scores.max() + 1e-9))
    
    boost = df.iloc[cand_ids]['vote_average'].values * np.log1p(df.iloc[cand_ids]['vote_count'].values)
    final_scores = (b_scores + v_scores) * boost
    
    top_idx = cand_ids[np.argsort(final_scores)[-top_k:][::-1]]
    return df.iloc[top_idx][["title", "overview"]].to_dict("records")