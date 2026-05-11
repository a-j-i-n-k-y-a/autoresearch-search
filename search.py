import numpy as np
from rank_bm25 import BM25Okapi
import faiss

def search(query, df, bm25, model, index, top_k=10):
    tokenized = query.lower().split()
    query_vec = model.encode([query]).astype("float32")
    
    vec_distances, vec_ids = index.search(query_vec, top_k * 5)
    vec_ids = vec_ids[0]
    
    bm25_scores = bm25.get_scores(tokenized)
    bm25_cands = np.argsort(bm25_scores)[-50:]
    
    cand_ids = np.unique(np.concatenate([vec_ids[vec_ids != -1], bm25_cands]))
    
    bm25_s = bm25_scores[cand_ids]
    
    faiss_s = np.zeros(len(cand_ids))
    for i, cid in enumerate(cand_ids):
        idx = np.where(vec_ids == cid)[0]
        faiss_s[i] = vec_distances[0][idx[0]] if len(idx) > 0 else 100.0
        
    bm25_r = np.argsort(np.argsort(-bm25_s))
    faiss_r = np.argsort(np.argsort(faiss_s))
    
    rrf = 1.0 / (bm25_r + faiss_r + 60)
    
    boost = df.iloc[cand_ids]['vote_average'].values * np.log1p(df.iloc[cand_ids]['vote_count'].values)
    final_scores = rrf * boost
    
    top_idx = cand_ids[np.argsort(final_scores)[-top_k:][::-1]]
    return df.iloc[top_idx][["title", "overview"]].to_dict("records")