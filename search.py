import numpy as np

def search(query, df, bm25, model, index, top_k=10):
    tokenized = query.lower().split()
    scores = bm25.get_scores(tokenized)
    top_idx = np.argsort(scores)[-top_k:][::-1]
    return df.iloc[top_idx][["title", "overview"]].to_dict("records")