import numpy as np

def search(query, df, bm25, model, index, top_k=10):
    # Expand query with genre signal if common keywords exist
    query_vec = model.encode([query])
    
    # Retrieve wider candidate pool
    distances, top_idx = index.search(query_vec.astype("float32"), top_k * 50)
    
    candidates = df.iloc[top_idx[0]].copy()
    candidates['dist'] = distances[0]
    
    # Calculate similarity scores
    sim = 1.0 / (1.0 + candidates['dist'])
    
    # Combine semantic similarity with popularity
    # Use log normalization for vote_count to dampen impact of massive outliers
    pop_score = np.log1p(candidates['vote_count']) * 0.05
    candidates['score'] = sim + pop_score
    
    # Apply genre boost: check if query words appear in genres
    query_terms = query.lower().split()
    for term in query_terms:
        candidates['score'] += candidates['genres'].str.lower().str.contains(term, regex=False).astype(float) * 0.1
        
    ranked = candidates.sort_values(by="score", ascending=False)
    return ranked.head(top_k)[["title", "overview"]].to_dict("records")