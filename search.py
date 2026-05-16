import numpy as np

def search(query, df, bm25, model, index, top_k=10):
    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)
    
    # Retrieve top 500 candidates via BM25
    candidate_indices = np.argsort(bm25_scores)[-500:]
    candidates = df.iloc[candidate_indices].copy()
    
    # Semantic scoring: embed query and use index
    query_vec = model.encode([query]).astype("float32")
    dists, idxs = index.search(query_vec, 2000)
    
    # Create semantic score map
    res_map = {idx: 1.0/(1.0+d) for idx, d in zip(idxs[0], dists[0])}
    
    # Map scores and compute final metric
    # Use normalized log-vote count and rating for popularity boost
    pop_score = np.log1p(candidates['vote_count']) * candidates['vote_average']
    pop_score = (pop_score - pop_score.mean()) / (pop_score.std() + 1e-6)
    
    semantic_scores = candidates.index.map(lambda i: res_map.get(i, 0.0))
    candidates['score'] = semantic_scores + (0.1 * pop_score)
    
    return candidates.sort_values("score", ascending=False).head(top_k).to_dict("records")