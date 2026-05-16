import numpy as np

def search(query, df, bm25, model, index, top_k=10):
    query_vec = model.encode([query]).astype("float32")
    dists, idxs = index.search(query_vec, 200)
    
    candidates = df.iloc[idxs[0]].copy()
    
    # Log-popularity boost: log(1 + vote_count)
    # This prevents extreme outliers from dominating
    popularity = np.log1p(candidates['vote_count'])
    
    # L2 distance is smaller for better matches, so subtract it
    # Use negative L2 distance to map to a standard "higher is better" scale
    scores = -dists[0] + (popularity * 0.5)
    
    candidates['score'] = scores
    return candidates.sort_values("score", ascending=False).head(top_k).to_dict("records")