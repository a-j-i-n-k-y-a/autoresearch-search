import numpy as np

def search(query, df, bm25, model, index, top_k=10):
    query_vec = model.encode([query])
    distances, top_idx = index.search(query_vec.astype("float32"), top_k * 100)
    
    candidates = df.iloc[top_idx[0]].copy()
    candidates['dist'] = distances[0]
    
    # Use normalized vote_average (1-10) and log-normalized vote_count
    # to bias towards quality and popularity
    norm_vote = (candidates['vote_average'] / 10.0)
    norm_count = np.log1p(candidates['vote_count']) / np.log1p(df['vote_count'].max())
    
    # Semantic score + metadata bias
    # Higher weights to semantic distance
    candidates['score'] = (1.0 / (1.0 + candidates['dist'])) + (norm_vote * 0.2) + (norm_count * 0.1)
    
    ranked = candidates.sort_values(by="score", ascending=False)
    return ranked.head(top_k)[["title", "overview"]].to_dict("records")