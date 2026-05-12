import numpy as np

def search(query, df, bm25, model, index, top_k=10):
    vec = model.encode([query])
    # Fetch a larger candidate pool to improve precision
    _, top_idx = index.search(vec.astype("float32"), 100)
    
    # Filter candidates by vote count to prioritize popular/verified entries
    candidates = df.iloc[top_idx[0]]
    ranked = candidates.sort_values(by="vote_count", ascending=False)
    
    return ranked.head(top_k)[["title", "overview"]].to_dict("records")