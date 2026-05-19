import numpy as np
from rank_bm25 import BM25Okapi
import re

def _bm25_tokenize(text):
    return re.findall(r'\b\w\w+\b', text.lower())

def search(query, df, bm25, model, index, top_k=10):
    tokens = _bm25_tokenize(query)
    bm25_scores = bm25.get_scores(tokens)
    
    query_vec = model.encode([query]).astype("float32")
    dists, idx_s = index.search(query_vec, 200)
    
    k = 60
    rrf_scores = np.zeros(len(df))
    
    # Efficient BM25 retrieval
    idx_b = np.argsort(bm25_scores)[-200:]
    for rank, i in enumerate(idx_b[::-1]):
        rrf_scores[i] += 1.0 / (k + rank)
        
    # Semantic retrieval
    for rank, i in enumerate(idx_s[0]):
        rrf_scores[i] += 1.0 / (k + rank)
        
    # Genre boost: identify candidates and apply mask
    candidates = np.where(rrf_scores > 0)[0]
    query_genres = [t for t in tokens if t in ["action", "comedy", "drama", "horror", "sci-fi", "thriller", "romance", "western", "crime"]]
    
    if query_genres:
        genres_col = df['genres'].values
        for i in candidates:
            genre_str = str(genres_col[i]).lower()
            if any(g in genre_str for g in query_genres):
                rrf_scores[i] *= 1.5
                
    # Popularity bias
    rrf_scores[candidates] += np.log1p(df['vote_count'].iloc[candidates].values) * 0.01
    
    top_indices = np.argsort(rrf_scores)[-top_k:][::-1]
    
    return df.iloc[top_indices].to_dict("records")