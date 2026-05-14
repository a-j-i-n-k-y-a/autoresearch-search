import numpy as np

def search(query, df, bm25, model, index, top_k=10):
    query_vec = model.encode([query]).astype("float32")
    _, idxs = index.search(query_vec, top_k)
    return df.iloc[idxs[0]].to_dict("records")