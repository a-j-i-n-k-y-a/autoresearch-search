import numpy as np

def search(query, df, bm25, model, index, top_k=10):
    query_vec = model.encode([query])
    # Search 500 candidates for wider retrieval
    distances, top_idx = index.search(query_vec.astype("float32"), 500)
    
    candidates = df.iloc[top_idx[0]].copy()
    # Normalize L2 distance to [0, 1]
    sim = 1.0 - (distances[0] / (1.0 + distances[0]))
    
    # Metadata normalization: log-scale popularity and normalize rating
    pop = np.log1p(candidates['vote_count'])
    pop = (pop - pop.min()) / (pop.max() - pop.min() + 1e-9)
    rating = candidates['vote_average'] / 10.0
    
    # Combined score
    candidates['score'] = sim + (0.15 * pop) + (0.15 * rating)
    
    ranked = candidates.sort_values(by="score", ascending=False)
    return ranked.head(top_k)[["title", "overview"]].to_dict("records")