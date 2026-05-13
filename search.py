import numpy as np

def search(query, df, bm25, model, index, top_k=10):
    query_vec = model.encode([query])
    # Search a slightly larger pool to allow for popularity re-ranking
    distances, top_idx = index.search(query_vec.astype("float32"), top_k * 5)
    
    candidates = df.iloc[top_idx[0]].copy()
    # Normalize distances (L2 distance is non-negative)
    dist = distances[0]
    # Log-transform vote count to scale popularity impact
    pop = np.log1p(candidates['vote_count'])
    # Combine: minimize distance, maximize log-popularity
    # Use a small divisor for popularity to balance against distance scale
    score = (1.0 / (1.0 + dist)) + (pop * 0.05)
    
    candidates['score'] = score
    ranked = candidates.sort_values(by="score", ascending=False)
    return ranked.head(top_k)[["title", "overview"]].to_dict("records")