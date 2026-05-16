import numpy as np

def search(query, df, bm25, model, index, top_k=10):
    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)
    
    # Retrieve top 1000 candidates using BM25 to get a strong initial recall
    candidate_indices = np.argsort(bm25_scores)[-1000:]
    candidates = df.iloc[candidate_indices].copy()
    
    # Get query vector once
    query_vec = model.encode([query]).astype("float32")
    
    # Calculate semantic scores for the candidates
    # Extract existing vectors from the index for these candidates
    # To be efficient, we perform a search on the full index but limit to the subset
    # Or just use the model to encode the overview of these 1000 candidates
    # Given we have an index, let's use the index search but filter by our candidates
    # Actually, simpler: compute L2 distance via model.encode(candidates) if they were small,
    # but since index exists, we search it.
    
    # Semantic scoring: 1 / (1 + L2)
    # Re-using the index: we get full 1000, then filter
    dists, idxs = index.search(query_vec, 2000)
    
    # map index results to score
    res_map = {idx: 1.0/(1.0+d) for idx, d in zip(idxs[0], dists[0])}
    
    candidates['semantic'] = candidates.index.map(lambda i: res_map.get(i, 0.0))
    
    # Popularity boost: log1p(vote_count) * vote_average
    popularity = np.log1p(candidates['vote_count']) * candidates['vote_average']
    
    # Final score: mix of semantic and popularity
    candidates['score'] = candidates['semantic'] * (1.0 + 0.05 * popularity)
    
    return candidates.sort_values("score", ascending=False).head(top_k).to_dict("records")