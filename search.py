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

    # BM25 scores for the candidate set
    bm25_scores = bm25.get_scores(tokenized)
    cand_scores = [(doc_id, bm25_scores[doc_id]) for doc_id in cand_ids]

    # rerank by BM25 descending
    cand_scores.sort(key=lambda x: x[1], reverse=True)

    # take the best top_k indices
    top_indices = [doc_id for doc_id, _ in cand_scores[:top_k]]

    return df.iloc[top_indices][["title", "overview"]].to_dict("records")