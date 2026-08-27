"""part_e / pe05 - Tokenization: word vs subword (BPE) comparison.

Compares token counts/compression of (a) whitespace word tokenization, (b) a
learned BPE tokenizer via `tokenizers` (HuggingFace) when available, and (c) a
tiny from-scratch BPE implementation so the demo runs with no extra deps.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "figures")

TEXT = ("natural language processing enables transformers to understand text "
        "naturalisation unnatural languages processing natural")


def word_tokenize(s):
    return s.split()


def train_bpe(corpus, vocab_size=40):
    # minimal BPE over characters -> merges
    tokens = list(" ".join(corpus.split()))
    merges = []
    from collections import Counter
    while len(set(tokens)) + len(merges) < vocab_size:
        pairs = Counter(zip(tokens, tokens[1:]))
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        new = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best:
                new.append(tokens[i] + tokens[i + 1]); i += 2
            else:
                new.append(tokens[i]); i += 1
        tokens = new; merges.append(best)
    return tokens, merges


def main():
    words = word_tokenize(TEXT)
    print("word tokens:", len(words))
    bpe_tokens, _ = train_bpe(TEXT)
    print("BPE tokens (char-merged):", len(bpe_tokens))
    try:
        from tokenizers import Tokenizer, models, trainers
        tk = Tokenizer(models.BPE())
        tk.train_from_iterator([TEXT], trainers.BpeTrainer(vocab_size=50))
        hf = tk.encode(TEXT).tokens
        print("HuggingFace BPE tokens:", len(hf))
        counts = [len(words), len(bpe_tokens), len(hf)]
        labels = ["word", "from-scratch BPE", "HF BPE"]
    except Exception as e:
        print("[pe05] tokenizers not installed; using from-scratch BPE only.")
        counts = [len(words), len(bpe_tokens)]; labels = ["word", "from-scratch BPE"]

    plt.figure(); plt.bar(labels, counts)
    plt.title("pe05 tokenization: fewer tokens = better compression")
    plt.ylabel("token count"); plt.tight_layout(); plt.savefig(f"{FIG}/pe05_tokens.png"); plt.close()
    print("Figures:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
