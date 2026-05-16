import numpy as np

def search(query, df, bm25, model, index, top_k=10):
    query_vec = model.encode([query]).astype("float32")
    dists, idxs = index.search(query_vec, 200)
    
    candidates = df.iloc[idxs[0]].copy()
    popularity = np.log1p(candidates['vote_count'])
    
    # Use negative L2 distance (higher is better) combined with popularity
    # Normalizing distance by 10.0 to balance with log-popularity range
    candidates['score'] = -dists[0] + (popularity * 0.5)
    
    return candidates.sort_values("score", ascending=False).head(top_k).to_dict("records")