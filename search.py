import numpy as np

def search(query, df, bm25, model, index, top_k=10):
    bm25_scores = bm25.get_scores(query.lower().split())
    top_n = 50
    candidates_idx = np.argsort(bm25_scores)[-top_n:][::-1]
    
    candidates = df.iloc[candidates_idx].copy()
    
    query_vec = model.encode([query])
    candidate_vecs = model.encode(candidates['text'].tolist())
    
    similarities = np.dot(candidate_vecs, query_vec.T).flatten()
    
    candidates['score'] = similarities
    ranked = candidates.sort_values(by="score", ascending=False)
    
    return ranked.head(top_k)[["title", "overview"]].to_dict("records")