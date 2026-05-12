import numpy as np

def search(query, df, bm25, model, index, top_k=10):
    vec = model.encode([query])
    distances, top_idx = index.search(vec.astype("float32"), top_k * 10)
    
    candidates = df.iloc[top_idx[0]].copy()
    candidates['dist'] = distances[0]
    
    # Use L2 distance (lower is better) and normalize vote_count
    # Invert distances to turn into a similarity score
    candidates['score'] = (1.0 / (1.0 + candidates['dist'])) * (1.0 + np.log1p(candidates['vote_count']) * 0.1)
    
    ranked = candidates.sort_values(by="score", ascending=False)
    
    return ranked.head(top_k)[["title", "overview"]].to_dict("records")