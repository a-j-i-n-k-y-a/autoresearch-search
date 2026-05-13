import numpy as np

def search(query, df, bm25, model, index, top_k=10):
    query_vec = model.encode([query])
    distances, top_idx = index.search(query_vec.astype("float32"), top_k * 10)
    
    candidates = df.iloc[top_idx[0]].copy()
    
    # Semantic score
    sim = 1.0 / (1.0 + distances[0])
    
    # Boost by vote_average for quality filter, only if vote_count is significant
    # Using a simple multiplicative boost for higher quality entries
    popularity_boost = (candidates['vote_average'] / 10.0) * (np.log1p(candidates['vote_count']) / 10.0)
    
    candidates['score'] = sim + (0.2 * popularity_boost)
    
    ranked = candidates.sort_values(by="score", ascending=False)
    return ranked.head(top_k)[["title", "overview"]].to_dict("records")