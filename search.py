import numpy as np

def search(query, df, bm25, model, index, top_k=10):
    vec = model.encode([query])
    _, top_idx = index.search(vec.astype("float32"), top_k * 5)
    
    candidates = df.iloc[top_idx[0]].copy()
    
    # Genre boost: simple match check
    query_lower = query.lower()
    candidates['genre_boost'] = candidates['genres'].apply(
        lambda g: 1.2 if any(word in g.lower() for word in query_lower.split()) else 1.0
    )
    
    candidates['score'] = candidates['vote_count'].rank(pct=True) * candidates['genre_boost']
    ranked = candidates.sort_values(by="score", ascending=False)
    
    return ranked.head(top_k)[["title", "overview"]].to_dict("records")