import numpy as np

def search(query, df, bm25, model, index, top_k=10):
    query_vec = model.encode([query]).astype("float32")
    _, top_idx = index.search(query_vec, top_k)
    return df.iloc[top_idx[0]].to_dict("records")