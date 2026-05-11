import numpy as np

def search(query, df, bm25, model, index, top_k=10):
    tokenized = query.lower().split()
    scores = bm25.get_scores(tokenized)
    
    # Boost by popularity (vote_average * log1p(vote_count))
    boost = df['vote_average'].values * np.log1p(df['vote_count'].values)
    final_scores = scores + (boost * 0.05)
    
    top_idx = np.argsort(final_scores)[-top_k:][::-1]
    return df.iloc[top_idx][["title", "overview"]].to_dict("records")