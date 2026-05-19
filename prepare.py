# prepare.py
# Fixed constants, data prep, and evaluation
# DO NOT MODIFY — this is the ground truth

import math
import numpy as np
import pandas as pd
import time
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle
import re
import unicodedata
import random

# ─── FIXED CONSTANTS ───────────────────────────────────────
DATASET_SIZE    = 722796
TOP_K           = 10
CACHE_DIR       = "data/"
EXPERIMENT_TIME = 60

# ─── TITLE NORMALIZATION ──────────────────────────────────

TITLE_ALIASES = {

    # article normalization
    "conjuring": "the conjuring",

    # franchise/entity normalization
    "indiana jones": "raiders of the lost ark",

    # canonicalized variants
    "romeo and juliet": "romeo juliet",

    # punctuation variants
    "mission impossible": "mission impossible",
}


def normalize_title(title: str) -> str:
    """
    Deterministic title normalization for benchmark evaluation.
    """

    title = title.lower().strip()

    # remove accents
    title = unicodedata.normalize("NFKD", title)
    title = title.encode("ascii", "ignore").decode("ascii")

    # remove punctuation
    title = re.sub(r"[^a-z0-9]+", " ", title)

    # collapse whitespace
    title = " ".join(title.split())

    # alias mapping
    title = TITLE_ALIASES.get(title, title)

    return title

# ─── BM25 TOKENIZER ──────────────────────────────────────
# Used consistently in prepare_data() and load_resources().
# IMPORTANT: search.py must replicate this logic when building
# BM25 query tokens, since it cannot import from prepare.py.
# Mismatch is non-fatal (unknown tokens score 0) but suboptimal.

_BM25_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "it", "its", "as",
    "be", "was", "are", "been", "being", "have", "has", "had",
    "do", "does", "did", "that", "this", "they", "their", "them",
    "he", "she", "we", "you", "i", "me", "my", "his", "her", "our",
    "will", "would", "could", "should", "may", "might", "can",
    "not", "no", "so", "if", "then", "than", "when", "where",
    "who", "which", "what", "how", "about", "up", "out", "into",
    "all", "more", "also", "just", "one", "two", "three", "some",
    "after", "before", "between", "through", "during", "because",
})


def _bm25_tokenize(text: str) -> list:
    """
    Normalizing tokenizer for BM25 — more aggressive than raw .split().

    Pipeline:
      1. Accent removal       ("léon" → "leon")
      2. Lowercase
      3. Punctuation → space  ("mission: impossible" → "mission impossible")
      4. Split
      5. Stopword removal     (drops "the", "a", "of", …)
      6. Length filter        (drops single-char noise tokens)

    Stemming is intentionally omitted: it requires nltk (not installed)
    and the marginal gain over steps 1–6 is small for movie titles/overviews.
    Add PorterStemmer here if you install nltk later.
    """
    # 1. accent removal
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    # 2. lowercase
    text = text.lower()
    # 3. punctuation → space
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # 4–6. split, filter
    return [t for t in text.split() if t not in _BM25_STOPWORDS and len(t) > 1]


# ─── RANK-AWARE METRIC HELPERS ────────────────────────────
# Pure functions — no I/O, no state.
# All take a ranked list of normalized titles and a set of normalized
# expected titles. Position 0 = rank 1.

def _mrr_at_k(ranked_titles: list, relevant: set, k: int) -> float:
    """
    Mean Reciprocal Rank at k.
    Returns 1/rank of the first relevant result, or 0 if none in top k.
    With multiple expected titles, rewards the earliest hit.
    """
    for i, title in enumerate(ranked_titles[:k]):
        if title in relevant:
            return 1.0 / (i + 1)
    return 0.0


def _ndcg_at_k(ranked_titles: list, relevant: set, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at k with binary relevance.
    DCG discount: 1 / log2(rank + 1), so rank 1 = 1.0, rank 2 ≈ 0.63, …
    Ideal DCG places all relevant docs at the top consecutive positions.
    Returns 0 if there are no relevant documents.
    """
    dcg = sum(
        1.0 / math.log2(i + 2)          # i+2: log2(pos+1), pos is 1-indexed
        for i, title in enumerate(ranked_titles[:k])
        if title in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0.0 else 0.0


def _precision_at_k(ranked_titles: list, relevant: set, k: int) -> float:
    """
    Precision at k — fraction of the top-k results that are relevant.
    Penalizes returning many irrelevant results even if recall is high.
    """
    if k == 0:
        return 0.0
    hits = sum(1 for t in ranked_titles[:k] if t in relevant)
    return hits / k


def _top1_hit(ranked_titles: list, relevant: set) -> float:
    """
    1.0 if the top-ranked result is relevant, 0.0 otherwise.
    Measures immediate usefulness — did the user get what they wanted first?
    """
    if not ranked_titles:
        return 0.0
    return 1.0 if ranked_titles[0] in relevant else 0.0


# ─── BENCHMARK QUERIES ───────────────────────────────────

BENCHMARK_QUERIES = [

    # ───────────────── TRAIN (25) ─────────────────

    {
        "query": "dream heist movie Leonardo DiCaprio layers of subconscious",
        "expected": ["Inception"],
        "slice": "semantic",
        "split": "train"
    },
    {
        "query": "astronaut stranded in space wormhole black hole",
        "expected": ["Interstellar", "Gravity"],
        "slice": "semantic",
        "split": "train"
    },
    {
        "query": "robot humanoid artificial intelligence consciousness",
        "expected": ["Ex Machina", "A.I. Artificial Intelligence"],
        "slice": "semantic",
        "split": "train"
    },
    {
        "query": "time machine going back to the future paradox",
        "expected": ["Back to the Future", "Looper"],
        "slice": "semantic",
        "split": "train"
    },
    {
        "query": "mafia family crime saga generational power",
        "expected": ["The Godfather", "Goodfellas"],
        "slice": "semantic",
        "split": "train"
    },
    {
        "query": "animated toys come to life friendship adventure",
        "expected": ["Toy Story"],
        "slice": "semantic",
        "split": "train"
    },
    {
        "query": "survival horror haunted hotel writer isolation",
        "expected": ["The Shining"],
        "slice": "semantic",
        "split": "train"
    },
    {
        "query": "boxing underdog champion training perseverance",
        "expected": ["Rocky", "Creed"],
        "slice": "genre",
        "split": "train"
    },
    {
        "query": "con artist identity fraud elaborate deception scheme",
        "expected": ["Catch Me If You Can", "The Sting"],
        "slice": "semantic",
        "split": "train"
    },
    {
        "query": "detective serial killer psychological cat and mouse",
        "expected": ["Se7en"],
        "slice": "semantic",
        "split": "train"
    },
    {
        "query": "alien invasion earth military resistance war",
        "expected": ["Independence Day", "War of the Worlds"],
        "slice": "genre",
        "split": "train"
    },
    {
        "query": "dystopian future society control rebellion uprising",
        "expected": ["The Matrix", "Divergent"],
        "slice": "semantic",
        "split": "train"
    },
    {
        "query": "time loop same day repeating stuck",
        "expected": ["Groundhog Day", "Edge of Tomorrow"],
        "slice": "semantic",
        "split": "train"
    },
    {
        "query": "virtual reality simulation video game trapped",
        "expected": ["Ready Player One", "Tron"],
        "slice": "semantic",
        "split": "train"
    },
    {
        "query": "prison escape wrongful conviction freedom hope",
        "expected": ["The Shawshank Redemption", "The Green Mile"],
        "slice": "semantic",
        "split": "train"
    },
    {
        "query": "stock market wall street greed financial corruption",
        "expected": ["The Wolf of Wall Street", "Margin Call"],
        "slice": "semantic",
        "split": "train"
    },
    {
        "query": "road trip self discovery friendship cross country",
        "expected": ["Little Miss Sunshine", "Into the Wild"],
        "slice": "genre",
        "split": "train"
    },
    {
        "query": "forbidden love star crossed romance tragedy",
        "expected": ["Romeo and Juliet", "Titanic"],
        "slice": "ambiguous",
        "split": "train"
    },
    {
        "query": "zombie apocalypse survival group undead outbreak",
        "expected": ["28 Days Later", "Dawn of the Dead"],
        "slice": "genre",
        "split": "train"
    },
    {
        "query": "supernatural demon possession exorcism religious horror",
        "expected": ["The Exorcist", "The Conjuring"],
        "slice": "genre",
        "split": "train"
    },
    {
        "query": "spy espionage secret agent government mission",
        "expected": ["Skyfall", "Mission: Impossible"],
        "slice": "ambiguous",
        "split": "train"
    },
    {
        "query": "treasure hunt ancient ruins archaeology adventure",
        "expected": ["Indiana Jones", "National Treasure"],
        "slice": "ambiguous",
        "split": "train"
    },
    {
        "query": "fish ocean lost family underwater adventure",
        "expected": ["Finding Nemo", "Finding Dory"],
        "slice": "genre",
        "split": "train"
    },
    {
        "query": "world war two jewish holocaust survival hiding",
        "expected": ["Schindler's List", "The Pianist"],
        "slice": "long_tail",
        "split": "train"
    },
    {
        "query": "tech startup silicon valley genius billionaire founder",
        "expected": ["The Social Network", "Jobs"],
        "slice": "semantic",
        "split": "train"
    },

    # ───────────────── DEV (15) ─────────────────

    {
        "query": "psychological thriller unreliable narrator mind bending twist",
        "expected": ["Shutter Island", "Black Swan"],
        "slice": "ambiguous",
        "split": "dev"
    },
    {
        "query": "superhero billionaire dark knight vigilante Gotham",
        "expected": ["The Dark Knight", "Batman Begins"],
        "slice": "exact_title",
        "split": "dev"
    },
    {
        "query": "world war two soldiers beach landing Normandy",
        "expected": ["Saving Private Ryan", "Dunkirk"],
        "slice": "semantic",
        "split": "dev"
    },
    {
        "query": "teenage high school prom coming of age awkward",
        "expected": ["Superbad", "Mean Girls"],
        "slice": "genre",
        "split": "dev"
    },
    {
        "query": "princess fairy tale kingdom true love musical",
        "expected": ["Beauty and the Beast", "Cinderella"],
        "slice": "genre",
        "split": "dev"
    },
    {
        "query": "heist bank robbery crew planning elaborate escape",
        "expected": ["Heat", "The Italian Job"],
        "slice": "genre",
        "split": "dev"
    },
    {
        "query": "space western bounty hunter frontier planets",
        "expected": ["Guardians of the Galaxy", "Serenity"],
        "slice": "ambiguous",
        "split": "dev"
    },
    {
        "query": "lawyer courtroom trial justice wrongful conviction",
        "expected": ["A Few Good Men", "12 Angry Men"],
        "slice": "genre",
        "split": "dev"
    },
    {
        "query": "hitman assassin contract killing redemption",
        "expected": ["Léon: The Professional", "John Wick"],
        "slice": "semantic",
        "split": "dev"
    },
    {
        "query": "drug cartel kingpin empire rise and fall",
        "expected": ["Scarface", "Traffic"],
        "slice": "semantic",
        "split": "dev"
    },
    {
        "query": "cloning genetic engineering identity human experiment",
        "expected": ["Never Let Me Go", "The Island"],
        "slice": "long_tail",
        "split": "dev"
    },
    {
        "query": "mental illness bipolar schizophrenia family struggle",
        "expected": ["A Beautiful Mind", "Silver Linings Playbook"],
        "slice": "semantic",
        "split": "dev"
    },
    {
        "query": "slasher masked killer teenagers summer camp",
        "expected": ["Friday the 13th", "Halloween"],
        "slice": "genre",
        "split": "dev"
    },
    {
        "query": "superhero team assemble save world threat",
        "expected": ["The Avengers", "Justice League"],
        "slice": "genre",
        "split": "dev"
    },
    {
        "query": "lion cub king pride savanna betrayal",
        "expected": ["The Lion King"],
        "slice": "exact_title",
        "split": "dev"
    },

    # ───────────────── TEST (10) ─────────────────

    {
        "query": "long distance relationship reunited second chance love",
        "expected": ["The Notebook", "Sleepless in Seattle"],
        "slice": "genre",
        "split": "test"
    },
    {
        "query": "romantic comedy misunderstanding enemies to lovers",
        "expected": ["When Harry Met Sally", "10 Things I Hate About You"],
        "slice": "genre",
        "split": "test"
    },
    {
        "query": "found footage paranormal haunting home camera",
        "expected": ["Paranormal Activity", "The Blair Witch Project"],
        "slice": "long_tail",
        "split": "test"
    },
    {
        "query": "martial arts kung fu master student training",
        "expected": ["The Karate Kid", "Crouching Tiger Hidden Dragon"],
        "slice": "genre",
        "split": "test"
    },
    {
        "query": "monsters children scream factory parallel world",
        "expected": ["Monsters, Inc."],
        "slice": "semantic",
        "split": "test"
    },
    {
        "query": "civil rights movement racism segregation protest",
        "expected": ["Selma", "42"],
        "slice": "long_tail",
        "split": "test"
    },
    {
        "query": "NASA space race moon landing engineers scientists",
        "expected": ["Hidden Figures", "First Man"],
        "slice": "semantic",
        "split": "test"
    },
    {
        "query": "wedding chaos family reunion dysfunction humor",
        "expected": ["Four Weddings and a Funeral"],
        "slice": "long_tail",
        "split": "test"
    },
    {
        "query": "office workplace boss employee absurd humor",
        "expected": ["Office Space", "Horrible Bosses"],
        "slice": "genre",
        "split": "test"
    },
    {
        "query": "musician rockstar rise fame addiction downfall",
        "expected": ["Bohemian Rhapsody", "Almost Famous"],
        "slice": "semantic",
        "split": "test"
    },
]


# ─── HOLDOUT QUERIES (never shown to LLM) ─────────────────

HOLDOUT_QUERIES = [

    {
        "query": "memory loss detective tattoos revenge mystery",
        "expected": ["Memento"],
        "slice": "semantic",
        "split": "holdout"
    },
    {
        "query": "submarine nuclear war ocean military tension",
        "expected": ["The Hunt for Red October"],
        "slice": "long_tail",
        "split": "holdout"
    },
    {
        "query": "journalist uncovers government conspiracy corruption coverup",
        "expected": ["Spotlight", "All the President's Men"],
        "slice": "semantic",
        "split": "holdout"
    },
    {
        "query": "desert post apocalyptic car chase survival wasteland",
        "expected": ["Mad Max: Fury Road"],
        "slice": "semantic",
        "split": "holdout"
    },
    {
        "query": "chef cooking restaurant food family pressure",
        "expected": ["Chef", "Ratatouille"],
        "slice": "genre",
        "split": "holdout"
    },
    {
        "query": "winter snow isolated creature paranoia horror",
        "expected": ["The Thing"],
        "slice": "long_tail",
        "split": "holdout"
    },
    {
        "query": "computer hacker cybercrime digital surveillance",
        "expected": ["The Girl with the Dragon Tattoo", "Hackers"],
        "slice": "ambiguous",
        "split": "holdout"
    },
    {
        "query": "magician rivalry obsession dangerous competition",
        "expected": ["The Prestige"],
        "slice": "semantic",
        "split": "holdout"
    },
    {
        "query": "small town sheriff serial murders investigation",
        "expected": ["Fargo", "Mystic River"],
        "slice": "ambiguous",
        "split": "holdout"
    },
    {
        "query": "journal writing teenager depression emotional isolation",
        "expected": ["The Perks of Being a Wallflower"],
        "slice": "long_tail",
        "split": "holdout"
    },
    {
        "query": "ocean disaster ship sinking survival romance",
        "expected": ["Titanic", "Poseidon"],
        "slice": "ambiguous",
        "split": "holdout"
    },
]


# ─── DATA PREP (run once) ──────────────────────────────────

def prepare_data():
    os.makedirs(CACHE_DIR, exist_ok=True)

    # ── Dataset ───────────────────────────────────────
    if not os.path.exists(f"{CACHE_DIR}movies.pkl"):
        print("Downloading dataset...")

        dataset = load_dataset("wykonos/movies", split="train")
        df = pd.DataFrame(dataset)

        df = df[
            ["title", "overview", "genres", "vote_average", "vote_count"]
        ].dropna()

        df = df[df["vote_count"] > 100]
        df = df[df["vote_average"] > 5.0]

        df = df.head(DATASET_SIZE).reset_index(drop=True)

        df["text"] = (
            df["title"] + " "
            + df["overview"] + " "
            + df["genres"]
        )

        df.to_pickle(f"{CACHE_DIR}movies.pkl")

        print(f"Saved {len(df)} movies")

    else:
        print("Loading cached dataset...")
        df = pd.read_pickle(f"{CACHE_DIR}movies.pkl")

    # ── BM25 Cache ────────────────────────────────────
    if not os.path.exists(f"{CACHE_DIR}bm25.pkl"):
        print("Building BM25 index...")

        tokenized_corpus = [
            _bm25_tokenize(text)          # ← was: text.lower().split()
            for text in df["text"].tolist()
        ]

        bm25 = BM25Okapi(tokenized_corpus)

        with open(f"{CACHE_DIR}bm25.pkl", "wb") as f:
            pickle.dump(bm25, f)

        print("BM25 saved")

    else:
        print("BM25 cache already exists")

    # ── FAISS + Embeddings ────────────────────────────
    if not os.path.exists(f"{CACHE_DIR}faiss.index"):
        print("Building vector index...")

        model = SentenceTransformer("BAAI/bge-small-en-v1.5")

        embeddings = model.encode(
            df["text"].tolist(),
            show_progress_bar=True
        )

        embeddings = np.array(embeddings).astype("float32")

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        faiss.write_index(index, f"{CACHE_DIR}faiss.index")

        with open(f"{CACHE_DIR}embeddings.pkl", "wb") as f:
            pickle.dump(embeddings, f)

        print("Vector index saved")

    else:
        print("FAISS index already exists")

    print("Data ready.")

    # ── Benchmark Validation ─────────────────────────

    dataset_titles = {
        normalize_title(t)
        for t in df["title"].tolist()
    }

    missing = []

    total_expected = 0

    for item in BENCHMARK_QUERIES:
        for expected in item["expected"]:
            total_expected += 1

            if normalize_title(expected) not in dataset_titles:
                missing.append(expected)

    print(
        f"Benchmark title coverage: "
        f"{total_expected - len(missing)} / {total_expected}"
    )

    if missing:
        print("\n⚠️ Missing benchmark titles:")

        for m in sorted(set(missing)):
            print(" -", m)

    # ── Holdout Validation ───────────────────────────
    # Checked separately so holdout gaps are clearly distinguished
    # from benchmark gaps in output.

    holdout_missing = []
    holdout_total   = 0

    for item in HOLDOUT_QUERIES:
        for expected in item["expected"]:
            holdout_total += 1

            if normalize_title(expected) not in dataset_titles:
                holdout_missing.append(expected)

    print(
        f"Holdout title coverage:   "
        f"{holdout_total - len(holdout_missing)} / {holdout_total}"
    )

    if holdout_missing:
        print("\n⚠️ Missing holdout titles:")

        for m in sorted(set(holdout_missing)):
            print(" -", m)

# ─── LOAD RESOURCES ───────────────────────────────────────

def load_resources():
    df = pd.read_pickle(f"{CACHE_DIR}movies.pkl")

    # ── BM25 — load from cache, don't rebuild ─────────────
    bm25_path = f"{CACHE_DIR}bm25.pkl"
    if os.path.exists(bm25_path):
        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)
    else:
        # Fallback: prepare_data() wasn't run or pkl was deleted.
        # Build in-memory but warn — caller should run prepare_data() first.
        print("⚠️  bm25.pkl not found — rebuilding (run prepare_data() to cache)")
        bm25 = BM25Okapi([_bm25_tokenize(t) for t in df["text"].tolist()])

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    index = faiss.read_index(f"{CACHE_DIR}faiss.index")

    return df, bm25, model, index


# ─── EVALUATION (ground truth metric, never change) ────────

def _run_retrieval_pass(search_fn, df, bm25, model, index, split, sample_size=None, seed=None):
    """
    Single retrieval pass over the filtered query set.
    Centralises the three things every eval function used to repeat:
      1. query filtering   — BENCHMARK_QUERIES filtered by split, done once
      2. search_fn call    — one call per query, result stored in record
      3. normalization     — expected titles and result titles normalized once

    Returns a list of records, one per query:
        {
          "query":    str,
          "expected": set[str],   # normalize_title applied
          "ranked":   list[str],  # normalize_title applied, order preserved
          "slice":    str,
        }

    All public eval functions call this and then apply their own
    aggregation logic — they never touch search_fn or normalize_title directly.
    """
    if split == "all":
        queries = BENCHMARK_QUERIES
    else:
        queries = [q for q in BENCHMARK_QUERIES if q.get("split") == split]

    if not queries:
        raise ValueError(
            f"No queries found for split='{split}'. "
            f"Check that BENCHMARK_QUERIES have 'split' fields assigned."
        )
    
    # ── Random sub-sampling ────────────────────────────────
    # Used during gate eval to reduce benchmark memorization pressure.
    # Warmup and latency trials always use the full set (sample_size=None).
    if sample_size is not None and sample_size < len(queries):
        rng     = random.Random(seed)
        queries = rng.sample(queries, sample_size)

    records = []
    for item in queries:
        raw    = search_fn(item["query"], df, bm25, model, index, top_k=TOP_K)
        # AFTER
        seen   = set()
        ranked = []
        for r in raw:
            t = normalize_title(r["title"])
            if t not in seen:
                seen.add(t)
                ranked.append(t)

        records.append({
            "query":    item["query"],
            "expected": {normalize_title(e) for e in item["expected"]},
            "ranked":   ranked,
            "slice":    item.get("slice", "unknown"),
        })
    return records

# ─── METRIC COMPUTERS (take pre-computed records, no search_fn) ───
# Called by run_full_eval(). Public eval functions are thin wrappers
# kept for backward compatibility with any direct callers.

def _compute_recall(records):
    total = sum(
        sum(1 for t in r["ranked"] if t in r["expected"]) / len(r["expected"])
        for r in records
    )
    return round(total / len(records), 6)


def _compute_metrics(records):
    totals = {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0, "precision": 0.0, "top1": 0.0}
    for r in records:
        ranked, expected = r["ranked"], r["expected"]
        found = sum(1 for t in ranked if t in expected)
        totals["recall"]    += found / len(expected)
        totals["mrr"]       += _mrr_at_k(ranked, expected, TOP_K)
        totals["ndcg"]      += _ndcg_at_k(ranked, expected, TOP_K)
        totals["precision"] += _precision_at_k(ranked, expected, TOP_K)
        totals["top1"]      += _top1_hit(ranked, expected)
    n = len(records)
    return {k: round(v / n, 6) for k, v in totals.items()}


def _compute_slices(records):
    slice_totals = {}
    slice_counts = {}
    for r in records:
        found  = sum(1 for t in r["ranked"] if t in r["expected"])
        recall = found / len(r["expected"])
        s      = r["slice"]
        slice_totals[s] = slice_totals.get(s, 0.0) + recall
        slice_counts[s] = slice_counts.get(s, 0)   + 1
    return {s: round(slice_totals[s] / slice_counts[s], 6) for s in slice_totals}


def _compute_per_query(records):
    results = {}
    for r in records:
        ranked, expected = r["ranked"], r["expected"]
        found = sum(1 for t in ranked if t in expected)
        rank_of_first_hit = next(
            (i + 1 for i, t in enumerate(ranked) if t in expected), None
        )
        results[r["query"]] = {
            "recall":            round(found / len(expected), 6),
            "mrr":               round(_mrr_at_k(ranked, expected, TOP_K), 6),
            "ndcg":              round(_ndcg_at_k(ranked, expected, TOP_K), 6),
            "precision":         round(_precision_at_k(ranked, expected, TOP_K), 6),
            "top1":              round(_top1_hit(ranked, expected), 6),
            "rank_of_first_hit": rank_of_first_hit,
            "retrieved_titles":  ranked,
        }
    return results


def run_full_eval(search_fn, df, bm25, model, index, split="dev", sample_size=None, seed=None):
    """
    Single retrieval pass that produces every metric in one shot.

    Replaces calling evaluate() + evaluate_metrics() + evaluate_by_slice()
    + per_query_results() separately — those run retrieval 4× over the same
    queries. This runs it once and feeds the records into all four computers.

    Returns a dict with keys:
        recall       — float          (gate metric)
        full_metrics — dict           (recall, mrr, ndcg, precision, top1)
        slices       — dict           (slice_name → recall)
        per_query    — dict           (query → per-query metric dict)
    """
    records = _run_retrieval_pass(search_fn, df, bm25, model, index, split, sample_size=sample_size, seed=seed)
    return {
        "recall":       _compute_recall(records),
        "full_metrics": _compute_metrics(records),
        "slices":       _compute_slices(records),
        "per_query":    _compute_per_query(records),
    }
    

def evaluate(search_fn, df, bm25, model, index, split="dev"):
    """
    Run benchmark queries through search_fn.
    Returns recall@K — the single metric to optimize.
    Higher is better.

    split="train"  — training queries only (agent sees these in prompts)
    split="dev"    — dev queries only (used for keep/discard decisions)
    split="test"   — test queries only (final benchmark eval, never used in agent loop)
    split="all"    — all BENCHMARK_QUERIES (train + dev + test)
    Note: HOLDOUT_QUERIES are never passed here — use evaluate_holdout() instead.
    """
    records = _run_retrieval_pass(search_fn, df, bm25, model, index, split)
    return _compute_recall(records)


def evaluate_metrics(search_fn, df, bm25, model, index, split="dev"):
    """
    Full rank-aware evaluation across all metric dimensions.

    Unlike evaluate(), which returns a single recall float, this returns a
    dict of five averaged metrics so you can see the complete quality picture:

        recall@10    — fraction of expected titles found anywhere in top 10.
                       Rank-blind: position 10 counts the same as position 1.
        mrr@10       — mean reciprocal rank of the first relevant result.
                       High if relevant results tend to appear near rank 1.
        ndcg@10      — normalized discounted cumulative gain (rank-discounted).
                       Captures both relevance and ordering simultaneously.
        precision@10 — fraction of all top-10 results that are relevant.
                       Low precision = lots of irrelevant noise at the top.
        top1         — fraction of queries where rank-1 result is relevant.
                       The most user-facing signal: "did I get it first try?"

    evaluate() is kept unchanged for backward compatibility with agent_loop.py.
    Use evaluate_metrics() when you need the full picture.

    Accepts the same split= values as evaluate().
    """
    records = _run_retrieval_pass(search_fn, df, bm25, model, index, split)
    return _compute_metrics(records)


def evaluate_by_slice(search_fn, df, bm25, model, index, split="dev"):
    """
    Run benchmark queries filtered by split, broken down by slice.
    Returns dict of {slice_name: recall_score}.

    Useful for diagnosing WHERE recall is strong or weak:
        exact_title : 0.95
        semantic    : 0.65
        genre       : 0.70
        long_tail   : 0.40
        ambiguous   : 0.55
    """
    records = _run_retrieval_pass(search_fn, df, bm25, model, index, split)
    return _compute_slices(records)


def per_query_results(search_fn, df, bm25, model, index, split="dev"):
    """
    Per-query diagnostics for regression analysis and experiment inspection.

    Returns a dict keyed by query string. Each value is a dict of metrics
    so you can see exactly what changed between experiments, not just whether
    aggregate recall moved:

        {
          "recall":            float,       # fraction of expected titles in top K
          "mrr":               float,       # 1/rank of first relevant hit (0 if none)
          "ndcg":              float,       # rank-discounted relevance score
          "precision":         float,       # fraction of top-K that are relevant
          "top1":              float,       # 1.0 if rank-1 result is relevant
          "rank_of_first_hit": int | None,  # 1-indexed; None if nothing hit
          "retrieved_titles":  list[str],   # normalized top-K titles as returned
        }

    Used as input to regression_report().  That function defaults to comparing
    on "recall"; pass metric= to compare on any key above.
    """
    records = _run_retrieval_pass(search_fn, df, bm25, model, index, split)
    return _compute_per_query(records)


def regression_report(prev_results, curr_results, metric="recall"):
    """
    Compares two per_query_results dicts and reports what got better or worse.

    Args:
        prev_results — output of per_query_results() before the change
        curr_results — output of per_query_results() after the change
        metric       — which metric to compare on (default: "recall").
                       Any key from per_query_results() is valid:
                       "recall", "mrr", "ndcg", "precision", "top1".

    Returns:
        {
          "regressions":  list of {query, before, after, delta},  sorted worst-first
          "improvements": list of {query, before, after, delta},  sorted best-first
          "net":          int,   # len(improvements) - len(regressions)
          "metric":       str,   # which metric was compared
        }

    The output format is intentionally stable: agent_loop.py accesses
    r["before"], r["after"], r["delta"] which still work regardless of
    which metric is being compared.

    Backward-compatible with the old {query: float} format from per_query_results().
    """

    def _extract(entry, key):
        """Pull metric value from either a dict (new) or bare float (legacy)."""
        if isinstance(entry, dict):
            return entry.get(key, 0.0)
        # Legacy: per_query_results() returned {query: float} (recall only).
        # Treat the float as recall; other metrics unavailable.
        return float(entry)

    regressions  = []
    improvements = []

    for query, prev_entry in prev_results.items():
        prev_score = _extract(prev_entry, metric)
        curr_score = _extract(curr_results.get(query, 0.0), metric)

        if curr_score < prev_score:
            regressions.append({
                "query":  query,
                "before": prev_score,
                "after":  curr_score,
                "delta":  round(curr_score - prev_score, 6),
            })
        elif curr_score > prev_score:
            improvements.append({
                "query":  query,
                "before": prev_score,
                "after":  curr_score,
                "delta":  round(curr_score - prev_score, 6),
            })

    regressions.sort(key=lambda r: r["delta"])          # worst first
    improvements.sort(key=lambda r: r["delta"], reverse=True)  # best first

    return {
        "regressions":  regressions,
        "improvements": improvements,
        "net":          len(improvements) - len(regressions),
        "metric":       metric,
    }


def evaluate_holdout(search_fn, df, bm25, model, index):
    """
    Evaluate search_fn against HOLDOUT_QUERIES — the private set that is
    NEVER shown to the agent or included in prompts/experiment history.

    Call this only for final reporting after all optimization is complete.
    Calling it mid-run leaks signal and defeats the purpose of the holdout.

    Returns:
        (avg_metrics, per_slice)

        avg_metrics — dict with recall, mrr, ndcg, precision, top1
                      averaged across all holdout queries.
        per_slice   — {slice_name: {recall, mrr, ndcg, precision, top1}}
                      breakdown by query type for diagnostic reporting.
    """
    if not HOLDOUT_QUERIES:
        raise ValueError("HOLDOUT_QUERIES is empty.")

    totals       = {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0, "precision": 0.0, "top1": 0.0}
    slice_totals = {}   # slice_name → {metric → float}
    slice_counts = {}   # slice_name → int

    for item in HOLDOUT_QUERIES:
        query    = item["query"]
        expected = {normalize_title(e) for e in item["expected"]}
        slice_   = item.get("slice", "unknown")

        # AFTER
        raw  = search_fn(query, df, bm25, model, index, top_k=TOP_K)
        seen = set()
        ranked = []
        for r in raw:
            t = normalize_title(r["title"])
            if t not in seen:
                seen.add(t)
                ranked.append(t)

        found = sum(1 for t in ranked if t in expected)
        q_metrics = {
            "recall":    found / len(expected),
            "mrr":       _mrr_at_k(ranked, expected, TOP_K),
            "ndcg":      _ndcg_at_k(ranked, expected, TOP_K),
            "precision": _precision_at_k(ranked, expected, TOP_K),
            "top1":      _top1_hit(ranked, expected),
        }

        for k, v in q_metrics.items():
            totals[k] += v

        if slice_ not in slice_totals:
            slice_totals[slice_] = {k: 0.0 for k in totals}
            slice_counts[slice_] = 0
        for k, v in q_metrics.items():
            slice_totals[slice_][k] += v
        slice_counts[slice_] += 1

    n = len(HOLDOUT_QUERIES)
    avg_metrics = {k: round(v / n, 6) for k, v in totals.items()}
    per_slice   = {
        s: {k: round(slice_totals[s][k] / slice_counts[s], 6) for k in totals}
        for s in slice_totals
    }

    return avg_metrics, per_slice


if __name__ == "__main__":
    prepare_data()
    print("Setup complete. Run agent_loop.py to start experiments.")