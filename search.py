import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

def search(query, df, bm25, model, index, top_k=10):
    tokenized = query.lower().split()

    query_vec = model.encode([query]).astype("float32")
    vec_distances, vec_ids = index.search(query_vec, top_k * 20)
    vec_ids = vec_ids[0]
    vec_distances = vec_distances[0]

    vec_cands = [int(i) for i in vec_ids if i != -1]
    vec_dist_cands = [d for i, d in zip(vec_ids, vec_distances) if i != -1]

    bm25_scores_all = bm25.get_scores(tokenized)
    bm25_top_idxs = np.argsort(bm25_scores_all)[::-1][: top_k * 20]
    bm25_cands = bm25_top_idxs.tolist()
    bm25_cand_scores = bm25_scores_all[bm25_top_idxs]

    cand_set = set(vec_cands) | set(bm25_cands)
    cand_ids = list(cand_set)

    bm25_cand_scores = np.array([bm25_scores_all[i] for i in cand_ids])

    title_bm25 = BM25Okapi(df['title'].apply(lambda x: x.lower().split()))
    title_bm25_scores_all = title_bm25.get_scores(tokenized)
    title_bm25_cand_scores = np.array([title_bm25_scores_all[i] for i in cand_ids])

    vec_id_to_dist = {cid: d for cid, d in zip(vec_cands, vec_dist_cands)}
    max_dist = max(vec_dist_cands) if vec_dist_cands else 1.0
    faiss_distances = np.array([vec_id_to_dist.get(i, max_dist) for i in cand_ids])

    bm25_ranks = np.argsort(np.argsort(-bm25_cand_scores))
    faiss_ranks = np.argsort(np.argsort(faiss_distances))

    rrf_scores = 1 / (bm25_ranks + faiss_ranks)

    vote_counts = df.iloc[cand_ids]['vote_count'].values
    vote_averages = df.iloc[cand_ids]['vote_average'].values
    rrf_scores *= np.log(vote_counts + 1) * vote_averages

    top_idx = np.argsort(rrf_scores)[::-1][:top_k]
    top_cand_ids = [cand_ids[i] for i in top_idx]

    return df.iloc[top_cand_ids][["title", "overview"]].to_dict("records")