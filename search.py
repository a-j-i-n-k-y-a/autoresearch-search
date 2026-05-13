import numpy as np

def search(query, df, bm25, model, index, top_k=10):
    query_vec = model.encode([query])
    _, top_idx = index.search(query_vec.astype("float32"), top_k)
    return df.iloc[top_idx[0]].to_dict("records")