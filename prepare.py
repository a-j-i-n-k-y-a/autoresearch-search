# prepare.py
# Fixed constants, data prep, and evaluation
# DO NOT MODIFY — this is the ground truth

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


# ─── BENCHMARK QUERIES (ground truth, never change) ────────

BENCHMARK_QUERIES = [
    {
        "query": "dream heist movie Leonardo DiCaprio layers of subconscious",
        "expected": ["Inception"]
    },
    {
        "query": "astronaut stranded in space wormhole black hole",
        "expected": ["Interstellar", "Gravity"]
    },
    {
        "query": "robot humanoid artificial intelligence consciousness",
        "expected": ["Ex Machina", "A.I. Artificial Intelligence"]
    },
    {
        "query": "psychological thriller unreliable narrator mind bending twist",
        "expected": ["Shutter Island", "Black Swan"]
    },
    {
        "query": "time machine going back to the future paradox",
        "expected": ["Back to the Future", "Looper"]
    },
    {
        "query": "mafia family crime saga generational power",
        "expected": ["The Godfather", "Goodfellas"]
    },
    {
        "query": "animated toys come to life friendship adventure",
        "expected": ["Toy Story"]
    },
    {
        "query": "superhero billionaire dark knight vigilante Gotham",
        "expected": ["The Dark Knight", "Batman Begins"]
    },
    {
        "query": "survival horror haunted hotel writer isolation",
        "expected": ["The Shining"]
    },
    {
        "query": "world war two soldiers beach landing Normandy",
        "expected": ["Saving Private Ryan", "Dunkirk"]
    },
    {
        "query": "teenage high school prom coming of age awkward",
        "expected": ["Superbad", "Mean Girls"]
    },
    {
        "query": "princess fairy tale kingdom true love musical",
        "expected": ["Beauty and the Beast", "Cinderella"]
    },
    {
        "query": "heist bank robbery crew planning elaborate escape",
        "expected": ["Heat", "The Italian Job"]
    },
    {
        "query": "boxing underdog champion training perseverance",
        "expected": ["Rocky", "Creed"]
    },
    {
        "query": "space western bounty hunter frontier planets",
        "expected": ["Guardians of the Galaxy", "Serenity"]
    },

    # CRIME / THRILLER
    {
        "query": "con artist identity fraud elaborate deception scheme",
        "expected": ["Catch Me If You Can", "The Sting"]
    },
    {
        "query": "detective serial killer psychological cat and mouse",
        "expected": ["Se7en"]
    },
    {
        "query": "lawyer courtroom trial justice wrongful conviction",
        "expected": ["A Few Good Men", "12 Angry Men"]
    },
    {
        "query": "hitman assassin contract killing redemption",
        "expected": ["Léon: The Professional", "John Wick"]
    },
    {
        "query": "drug cartel kingpin empire rise and fall",
        "expected": ["Scarface", "Traffic"]
    },

    # SCI-FI
    {
        "query": "alien invasion earth military resistance war",
        "expected": ["Independence Day", "War of the Worlds"]
    },
    {
        "query": "dystopian future society control rebellion uprising",
        "expected": ["The Matrix", "Divergent"]
    },
    {
        "query": "time loop same day repeating stuck",
        "expected": ["Groundhog Day", "Edge of Tomorrow"]
    },
    {
        "query": "virtual reality simulation video game trapped",
        "expected": ["Ready Player One", "Tron"]
    },
    {
        "query": "cloning genetic engineering identity human experiment",
        "expected": ["Never Let Me Go", "The Island"]
    },

    # DRAMA
    {
        "query": "prison escape wrongful conviction freedom hope",
        "expected": ["The Shawshank Redemption", "The Green Mile"]
    },
    {
        "query": "stock market wall street greed financial corruption",
        "expected": ["The Wolf of Wall Street", "Margin Call"]
    },
    {
        "query": "road trip self discovery friendship cross country",
        "expected": ["Little Miss Sunshine", "Into the Wild"]
    },
    {
        "query": "mental illness bipolar schizophrenia family struggle",
        "expected": ["A Beautiful Mind", "Silver Linings Playbook"]
    },
    {
        "query": "musician rockstar rise fame addiction downfall",
        "expected": ["Bohemian Rhapsody", "Almost Famous"]
    },

    # ROMANCE
    {
        "query": "forbidden love star crossed romance tragedy",
        "expected": ["Romeo and Juliet", "Titanic"]
    },
    {
        "query": "long distance relationship reunited second chance love",
        "expected": ["The Notebook", "Sleepless in Seattle"]
    },
    {
        "query": "romantic comedy misunderstanding enemies to lovers",
        "expected": ["When Harry Met Sally", "10 Things I Hate About You"]
    },

    # HORROR
    {
        "query": "zombie apocalypse survival group undead outbreak",
        "expected": ["28 Days Later", "Dawn of the Dead"]
    },
    {
        "query": "supernatural demon possession exorcism religious horror",
        "expected": ["The Exorcist", "The Conjuring"]
    },
    {
        "query": "slasher masked killer teenagers summer camp",
        "expected": ["Friday the 13th", "Halloween"]
    },
    {
        "query": "found footage paranormal haunting home camera",
        "expected": ["Paranormal Activity", "The Blair Witch Project"]
    },

    # ACTION / ADVENTURE
    {
        "query": "spy espionage secret agent government mission",
        "expected": ["Skyfall", "Mission: Impossible"]
    },
    {
        "query": "treasure hunt ancient ruins archaeology adventure",
        "expected": ["Indiana Jones", "National Treasure"]
    },
    {
        "query": "superhero team assemble save world threat",
        "expected": ["The Avengers", "Justice League"]
    },
    {
        "query": "martial arts kung fu master student training",
        "expected": ["The Karate Kid", "Crouching Tiger Hidden Dragon"]
    },

    # ANIMATION
    {
        "query": "fish ocean lost family underwater adventure",
        "expected": ["Finding Nemo", "Finding Dory"]
    },
    {
        "query": "lion cub king pride savanna betrayal",
        "expected": ["The Lion King"]
    },
    {
        "query": "monsters children scream factory parallel world",
        "expected": ["Monsters, Inc."]
    },
   

    # HISTORICAL / BIOGRAPHY
    {
        "query": "world war two jewish holocaust survival hiding",
        "expected": ["Schindler's List", "The Pianist"]
    },
    {
        "query": "civil rights movement racism segregation protest",
        "expected": ["Selma", "42"]
    },
    {
        "query": "NASA space race moon landing engineers scientists",
        "expected": ["Hidden Figures", "First Man"]
    },
    {
        "query": "tech startup silicon valley genius billionaire founder",
        "expected": ["The Social Network", "Jobs"]
    },

    # COMEDY
    {
        "query": "wedding chaos family reunion dysfunction humor",
        "expected": ["Four Weddings and a Funeral"]
    },
    {
        "query": "office workplace boss employee absurd humor",
        "expected": ["Office Space", "Horrible Bosses"]
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
            text.lower().split()
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

        model = SentenceTransformer("all-MiniLM-L6-v2")

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

# ─── LOAD RESOURCES ───────────────────────────────────────

def load_resources():
    df = pd.read_pickle(f"{CACHE_DIR}movies.pkl")

    texts = df["text"].tolist()

    bm25 = BM25Okapi(
        [t.lower().split() for t in texts]
    )

    model = SentenceTransformer("all-MiniLM-L6-v2")

    index = faiss.read_index(f"{CACHE_DIR}faiss.index")

    return df, bm25, model, index


# ─── EVALUATION (ground truth metric, never change) ────────

def evaluate(search_fn, df, bm25, model, index):
    """
    Run all benchmark queries through search_fn.
    Returns recall@K — the single metric to optimize.
    Higher is better.
    """

    total_recall = 0.0

    for item in BENCHMARK_QUERIES:

        query = item["query"]

        expected = {
            normalize_title(e)
            for e in item["expected"]
        }

        results = search_fn(
            query,
            df,
            bm25,
            model,
            index,
            top_k=TOP_K
        )

        retrieved = {
            normalize_title(r["title"])
            for r in results
        }

        found = sum(
            1 for e in expected
            if e in retrieved
        )

        recall = found / len(expected)

        total_recall += recall

    avg_recall = total_recall / len(BENCHMARK_QUERIES)

    return round(avg_recall, 6)


if __name__ == "__main__":
    prepare_data()
    print("Setup complete. Run agent_loop.py to start experiments.")