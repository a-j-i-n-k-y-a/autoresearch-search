import numpy as np
from rank_bm25 import BM25Okapi
import faiss

def search(query, df, bm25, model, index, top_k=10):
    tokenized = query.lower().split()
    query_vec = model.encode([query]).astype("float32")
    
    vec_distances, vec_ids = index.search(query_vec, top_k * 5)
    bm25_scores = bm25.get_scores(tokenized)
    
    bm25_cands = np.argsort(bm25_scores)[-50:]
    vec_cands = vec_ids[0]
    cand_ids = np.unique(np.concatenate([bm25_cands, vec_cands]))
    
    b_scores = bm25_scores[cand_ids]
    b_scores = (b_scores - b_scores.min()) / (b_scores.max() - b_scores.min() + 1e-9)
    
    v_scores = np.zeros(len(cand_ids))
    for i, cid in enumerate(cand_ids):
        match = np.where(vec_ids[0] == cid)[0]
        if len(match) > 0:
            v_scores[i] = 1.0 / (1.0 + vec_distances[0][match[0]])
            
    final_scores = b_scores + v_scores
    top_idx = cand_ids[np.argsort(final_scores)[-top_k:][::-1]]
    return df.iloc[top_idx][["title", "overview"]].to_dict("records")