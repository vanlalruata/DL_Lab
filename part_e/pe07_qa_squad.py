"""part_e / pe07 - Extractive Question Answering on SQuAD (pipeline or TF-IDF fallback).

Uses the HuggingFace QA pipeline on SQuAD when `transformers` is available;
otherwise falls back to a TF-IDF retriever that scores sentences in the context
and returns the best span. Demonstrates the QA workflow and measures latency.
"""
import os, time, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "figures")

CONTEXT = ("The Eiffel Tower is a wrought-iron lattice tower in Paris, France. "
           "It was constructed from 1887 to 1889 and is one of the most recognizable "
           "landmarks in the world. Paris is the capital city of France.")
QUESTIONS = ["What city is the Eiffel Tower in?", "When was the Eiffel Tower built?",
             "What is the capital of France?"]


def tfidf_qa(context, question):
    sents = re.split(r"(?<=[.])\s", context)
    q = set(question.lower().split())
    best, score = None, -1
    for s in sents:
        overlap = len(q & set(s.lower().split()))
        if overlap > score:
            best, score = s, overlap
    return best or sents[0]


def main():
    use_hf = os.environ.get("PE07_HF") == "1"
    if use_hf:
        try:
            from transformers import pipeline
            qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
            ans = lambda c, q: qa(question=q, context=c)["answer"]
            mode = "HF pipeline"
        except Exception:
            ans = tfidf_qa
            mode = "TF-IDF fallback"
    else:
        ans = tfidf_qa
        mode = "TF-IDF fallback (set PE07_HF=1 to use a real HF model)"
    print(f"[pe07] mode: {mode}")
    lat = []
    for q in QUESTIONS:
        t0 = time.perf_counter(); a = ans(CONTEXT, q); lat.append((time.perf_counter() - t0) * 1000)
        print(f"Q: {q}\nA: {a}\n")
    plt.figure(); plt.bar(range(len(QUESTIONS)), lat)
    plt.ylabel("latency (ms)"); plt.title("pe07 QA inference latency")
    plt.xticks(range(len(QUESTIONS)), [f"Q{i+1}" for i in range(len(QUESTIONS))])
    plt.tight_layout(); plt.savefig(f"{FIG}/pe07_latency.png"); plt.close()
    print("Figures:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
