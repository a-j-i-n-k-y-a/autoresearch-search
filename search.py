import numpy as np

def search(query, df, bm25, model, index, top_k=10):
    query_vec = model.encode([query])
    # Increase search pool to 100 to increase recall potential
    distances, top_idx = index.search(query_vec.astype("float32"), 100)
    
    candidates = df.iloc[top_idx[0]].copy()
    
    # Semantic score using inverted distance
    sim = 1.0 / (1.0 + distances[0])
    
    # Use log-scaled popularity boost to favor widely recognized/well-rated films
    popularity_boost = (candidates['vote_average'] / 10.0) * np.log1p(candidates['vote_count'])
    
    candidates['score'] = sim + (0.1 * popularity_boost)
    
    ranked = candidates.sort_values(by="score", ascending=False)
    return ranked.head(top_k)[["title", "overview"]].to_dict("records")