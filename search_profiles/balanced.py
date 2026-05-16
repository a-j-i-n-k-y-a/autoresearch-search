import numpy as np

def search(query, df, bm25, model, index, top_k=10):
    query_vec = model.encode([query]).astype("float32")
    dists, idxs = index.search(query_vec, 200)
    
    candidates = df.iloc[idxs[0]].copy()
    
    # Combined popularity and quality score
    pop = np.log1p(candidates['vote_count'])
    rating = candidates['vote_average']
    
    # Normalize L2 distance to [0, 1] range roughly
    norm_dist = 1.0 / (1.0 + dists[0])
    
    # Boost by popularity and rating (log scale)
    candidates['score'] = norm_dist * (1.0 + 0.1 * pop) * (1.0 + 0.1 * rating)
    
    return candidates.sort_values("score", ascending=False).head(top_k).to_dict("records")