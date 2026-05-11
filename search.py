import numpy as np

def search(query, df, bm25, model, index, top_k=10):
    tokenized = query.lower().split()
    scores = bm25.get_scores(tokenized)
    boost = np.log1p(df["vote_count"].values)
    final_scores = scores + 0.1 * boost
    top_idx = np.argsort(final_scores)[-top_k:][::-1]
    return df.iloc[top_idx][["title", "overview"]].to_dict("records")