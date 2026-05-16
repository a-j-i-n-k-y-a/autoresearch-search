import numpy as np

def search(query, df, bm25, model, index, top_k=10):
    query_vec = model.encode([query]).astype("float32")
    
    # Retrieve a wider pool to ensure high recall
    dists, idxs = index.search(query_vec, 500)
    
    candidates = df.iloc[idxs[0]].copy()
    
    # Semantic score: 1 / (1 + L2)
    semantic_score = 1.0 / (1.0 + dists[0])
    
    # Popularity metric: log1p(vote_count) * (vote_average / 10)
    # Using log1p helps dampen the effect of extreme vote counts
    popularity_score = np.log1p(candidates['vote_count']) * (candidates['vote_average'] / 10.0)
    
    # Combined score with a small popularity boost factor
    candidates['score'] = semantic_score * (1.0 + 0.15 * popularity_score)
    
    return candidates.sort_values("score", ascending=False).head(top_k).to_dict("records")