import numpy as np

def search(query, df, bm25, model, index, top_k=10):
    """
    Hybrid search: combine FAISS vector search and BM25 rankings.
    Retrieves candidates from both methods, then re‑ranks using a
    weighted sum of normalized scores.
    """
    tokenized = query.lower().split()

    # ---------- FAISS vector search ----------
    query_vec = model.encode([query]).astype("float32")
    vec_distances, vec_ids = index.search(query_vec, top_k * 5)
    vec_ids = vec_ids[0]
    vec_distances = vec_distances[0]

    # keep only valid ids
    vec_cands = [int(i) for i in vec_ids if i != -1]
    vec_dist_cands = [d for i, d in zip(vec_ids, vec_distances) if i != -1]

    # ---------- BM25 search ----------
    bm25_scores_all = bm25.get_scores(tokenized)
    bm25_top_idxs = np.argsort(bm25_scores_all)[::-1][: top_k * 5]
    bm25_cands = bm25_top_idxs.tolist()
    bm25_cand_scores = bm25_scores_all[bm25_top_idxs]

    # ---------- Union of candidates ----------
    cand_set = set(vec_cands) | set(bm25_cands)
    cand_ids = list(cand_set)

    # retrieve scores for each candidate
    # BM25 part
    bm25_cand_scores = np.array([bm25_scores_all[i] for i in cand_ids])

    # FAISS part: need to map candidate id to its distance if present
    vec_id_to_dist = {cid: d for cid, d in zip(vec_cands, vec_dist_cands)}
    # for missing ids use a large distance (worst similarity)
    max_dist = max(vec_dist_cands) if vec_dist_cands else 1.0
    faiss_distances = np.array([vec_id_to_dist.get(i, max_dist) for i in cand_ids])

    # ---------- Normalize scores ----------
    # BM25 normalization
    bm25_min, bm25_max = bm25_cand_scores.min(), bm25_cand_scores.max()
    bm25_norm = (bm25_cand_scores - bm25_min) / (bm25_max - bm25_min + 1e-8)

    # FAISS similarity normalization (convert distance to similarity)
    dist_min, dist_max = faiss_distances.min(), faiss_distances.max()
    faiss_sim = 1 - (faiss_distances - dist_min) / (dist_max - dist_min + 1e-8)

    # ---------- Combine ----------
    combined = bm25_norm + faiss_sim  # equal weighting; can adjust if needed
    top_idx = np.argsort(combined)[::-1][:top_k]
    top_cand_ids = [cand_ids[i] for i in top_idx]

    return df.iloc[top_cand_ids][["title", "overview"]].to_dict("records")