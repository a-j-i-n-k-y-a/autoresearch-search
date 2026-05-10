import numpy as np

def search(query, df, bm25, model, index, top_k=10):
    """
    Hybrid search: retrieve candidates with FAISS vector search,
    then rerank them using BM25 scores.
    """
    tokenized = query.lower().split()

    # vector search – fetch a larger pool (5 × top_k)
    query_vec = model.encode([query]).astype('float32')
    distances, vec_ids = index.search(query_vec, top_k * 5)

    # keep only valid document ids (ignore -1 placeholders)
    cand_ids = [int(i) for i in vec_ids[0] if i != -1]

    # if we got fewer candidates than needed, fall back to BM25 only
    if not cand_ids:
        scores = bm25.get_scores(tokenized)
        best = np.argsort(scores)[::-1][:top_k]
        return df.iloc[best][["title", "overview"]].to_dict("records")

    # get BM25 scores for candidates
    bm25_scores = bm25.get_scores(tokenized)

    # normalize and combine scores
    bm25_cand = np.array([bm25_scores[i] for i in cand_ids])
    dist_cand = distances[0][:len(cand_ids)]

    # convert distance to similarity (lower distance = higher similarity)
    bm25_norm = (bm25_cand - bm25_cand.min()) / (bm25_cand.max() - bm25_cand.min() + 1e-8)
    dist_norm = 1 - (dist_cand - dist_cand.min()) / (dist_cand.max() - dist_cand.min() + 1e-8)

    combined = bm25_norm + dist_norm
    top_idx = np.argsort(combined)[::-1][:top_k]
    top_cand_ids = [cand_ids[i] for i in top_idx]

    return df.iloc[top_cand_ids][["title", "overview"]].to_dict("records")