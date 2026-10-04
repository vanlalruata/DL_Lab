"""part_e / pe09 - Embeddings & semantic search (sentence-transformers or TF-IDF).

Encodes a small corpus into dense embeddings (sentence-transformers when available,
else a mean-pooled BiLSTM or TF-IDF vectors) and evaluates retrieval quality with
Recall@k using cosine similarity.
"""
import os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "figures")

CORPUS = [
    "the cat sat on the mat", "a dog ran in the park", "neural networks learn from data",
    "transformers use attention mechanisms", "the dog barked at the cat",
    "deep learning models need GPUs", "a cat slept on the sofa",
]
QUERIES = [("cat on mat", 0), ("dog and park", 1), ("attention in transformers", 3)]


def cos_sim(a, b):
    return a @ b.T / (np.linalg.norm(a, 1) * np.linalg.norm(b, 1) + 1e-9)


def main():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        E = model.encode(CORPUS); Q = model.encode([q for q, _ in QUERIES])
        mode = "sentence-transformers"
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer
        V = TfidfVectorizer().fit(CORPUS)
        E = V.transform(CORPUS).toarray(); Q = V.transform([q for q, _ in QUERIES]).toarray()
        mode = "TF-IDF fallback"
    print(f"[pe09] mode: {mode}")

    recalls = []
    for k in [1, 2, 3]:
        hits = 0
        for q, gold in QUERIES:
            qi = QUERIES.index((q, gold))
            sims = cos_sim(Q[qi:qi + 1], E)[0]
            top = np.argsort(-sims)[:k]
            if gold in top:
                hits += 1
        recalls.append(hits / len(QUERIES))
    print("Recall@k:", dict(zip([1, 2, 3], [round(r, 2) for r in recalls])))

    # show top-1 for each query
    for qi, (q, gold) in enumerate(QUERIES):
        top = np.argmax(cos_sim(Q[qi:qi + 1], E)[0])
        print(f"Q: {q!r} -> top1: {CORPUS[top]} (gold idx {gold})")

    plt.figure(); plt.plot([1, 2, 3], recalls, "o-"); plt.xlabel("k")
    plt.ylabel("Recall@k"); plt.title("pe09 retrieval quality"); plt.savefig(f"{FIG}/pe09_recall.png"); plt.close()
    print("Figures:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
