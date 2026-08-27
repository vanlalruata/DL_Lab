"""part_e / pe11 - Mizo corpus creation + POS tagging (tone/diacritic aware).

Builds a small parallel Mizo / English / Hindi corpus (seeded with the provided
examples), demonstrates a diacritic-aware tokenizer, and applies a lexicon POS
tagger. The key linguistic point: Mizo is tonal and diacritics change meaning and
part-of-speech, e.g.
    Lêi  (purchase)        -> V
    Léi  (tongue / bent)    -> N / ADJ
    Lèi  (bridge / ladder)  -> N
The corpus and POS tagger preserve these distinctions (lêi != léi != lèi).
"""
import os, json, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
DATA = os.path.join(HERE, "data")
os.makedirs(FIG, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

# (mizo, english, hindi) parallel sentences
CORPUS = [
    ("Damdawi ka lêi dawn", "I am going to purchase a medicine",
     "main davai khareedne jaa raha hoon"),
    ("I léi ka hmu thei", "I can see your tongue",
     "main tumhari jeebh dekh sakta hoon"),
    ("I va ding léi em aw", "I wonder why you stand bent",
     "main sochta hoon tum jhuke hue kyun khade ho"),
    ("Saw lèi saw kan kal thlen a ngai", "we need to reach that bridge",
     "hamein us pool tak pahuchna hi hai"),
    ("Ka lêi tha", "I bought a good one", "maine ek accha khareeda"),
    ("Léi a tawp", "the tongue is tired", "jeebh thak gayi"),
    ("Lèi chungah chuan", "on the bridge", "pool ke upar"),
    ("Nu leh pa te in lêi", "the parents purchase", "maa ba khareedte hain"),
    ("Zan ah Léi a sen", "at night the tongue is red", "raat me jeebh laal hai"),
    ("Lèi kha a sang", "the bridge is high", "pool ooncha hai"),
]

# Mizo lexicon for POS tagging (tone matters!)
MIZO_POS = {
    "ka": "PRON", "i": "PRON", "anu": "PRON", "i": "PRON", "kan": "PRON", "saw": "DET",
    "lêi": "VERB",      # purchase
    "léi": "NOUN",      # tongue / bent
    "lèi": "NOUN",      # bridge / ladder
    "damdawi": "NOUN", "ding": "VERB", "kal": "VERB", "hmu": "VERB", "thlen": "VERB",
    "dawn": "PART", "aw": "PART", "em": "PART", "chu": "PART", "chuan": "PART",
    "thu": "NOUN", "van": "ADJ", "sang": "ADJ", "thu": "NOUN", "vel": "NOUN",
    "nu": "NOUN", "pa": "NOUN", "leh": "CONJ", "a": "PRON", "khaw": "NOUN",
    "zan": "NOUN", "chungah": "ADP", "kha": "DET", "tawp": "ADJ", "ngai": "VERB",
}


def tokenize(text):
    # whitespace tokenization preserves diacritics (é/è/ê are distinct code points)
    return text.split()


def pos_tag_mizo(sentence):
    tags = []
    for w in tokenize(sentence):
        w = w.lower().rstrip(".,!?")
        tags.append((w, MIZO_POS.get(w, "NOUN")))
    return tags


def main():
    # 1) save corpus
    with open(os.path.join(DATA, "mizo_corpus.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mizo", "english", "hindi"])
        for m, e, h in CORPUS:
            w.writerow([m, e, h])
    with open(os.path.join(DATA, "mizo_corpus.json"), "w", encoding="utf-8") as f:
        json.dump(CORPUS, f, ensure_ascii=False, indent=2)
    print(f"Saved corpus ({len(CORPUS)} triples) to part_e/data/mizo_corpus.*")

    # 2) diacritic-aware tokenization check
    print("\nDiacritic-distinct tokens:")
    for tok in ["lêi", "léi", "lèi"]:
        print(f"  {tok!r} -> bytes={tok.encode('utf-8')}  POS={MIZO_POS.get(tok, 'NOUN')}")

    # 3) POS tagging on the example sentences
    print("\nPOS tagging examples:")
    for m, e, _ in CORPUS[:4]:
        print(f"  {m}")
        print("   ", pos_tag_mizo(m))

    # 4) visualize POS distribution
    from collections import Counter
    c = Counter(t for _, t in pos_tag_mizo(" ".join(m for m, _, _ in CORPUS)))
    plt.figure(); plt.bar(list(c.keys()), list(c.values()))
    plt.title("pe11 Mizo POS distribution"); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(f"{FIG}/pe11_pos_dist.png"); plt.close()

    # 5) show that dropping diacritics collapses the distinction
    def strip_diac(s):
        import unicodedata
        return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    print("\nDiacritic-stripped (loss of meaning):")
    for tok in ["lêi", "léi", "lèi"]:
        print(f"  {tok!r} -> {strip_diac(tok)!r}  (all become 'lei', POS ambiguous)")
    print("Figures:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
