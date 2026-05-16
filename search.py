import numpy as np

def search(query, df, bm25, model, index, top_k=10):
    query_vec = model.encode([query]).astype("float32")
    # Retrieve larger pool for re-ranking
    pool_size = 100
    dists, idxs = index.search(query_vec, pool_size)
    
    # Extract candidates
    candidates = df.iloc[idxs[0]].copy()
    
    # Normalize vote_count for popularity boost
    max_votes = df['vote_count'].max()
    popularity_boost = (candidates['vote_count'] / max_votes) * 0.1
    
    # Score = (inverted L2 distance) + popularity_boost
    # L2 distance is smaller for better matches
    scores = (1.0 / (1.0 + dists[0])) + popularity_boost
    
    candidates['score'] = scores
    return candidates.sort_values("score", ascending=False).head(top_k).to_dict("records")