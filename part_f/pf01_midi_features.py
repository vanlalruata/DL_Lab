"""part_f / pf01 - MIDI/WAV loading, feature extraction & EDA.

Builds (or loads) a multi-genre MIDI-style dataset, extracts hand-crafted features
(pitch/duration/velocity statistics), visualizes per-genre feature distributions,
writes a features CSV, and demonstrates WAV synthesis + spectrogram analysis.
Pure numpy/matplotlib/sklearn; optional pretty_midi for real MIDI I/O.
"""
import os, sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import music_utils as mu

FIG = os.path.join(os.path.dirname(__file__), "figures")
DATA = os.path.join(os.path.dirname(__file__), "data")
GEN = mu.GENRES


def main():
    os.makedirs(DATA, exist_ok=True)
    data = mu.download_or_load()  # synthetic by default; or pass a real midi dir
    feats = np.array([mu.extract_features(s) for s in data["seq"]])
    y = np.array(data["genre"])

    # save features csv
    with open(os.path.join(DATA, "features.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["genre"] + [f"f{i}" for i in range(feats.shape[1])])
        for g, row in zip(y, feats):
            w.writerow([g] + list(row))

    # per-feature boxplot per genre
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    flabels = ["pitch mean", "pitch std", "pitch range", "dur mean",
               "dur std", "vel mean", "tonal div", "step std"]
    for i, (ax, lab) in enumerate(zip(axes.ravel(), flabels)):
        vals = [feats[y == g, i] for g in GEN]
        ax.boxplot(vals, tick_labels=GEN)
        ax.set_title(lab)
        plt.setp(ax.get_xticklabels(), fontsize=7)
    plt.tight_layout(); plt.savefig(f"{FIG}/pf01_feature_boxplots.png"); plt.close()
    print("Feature means per genre:")
    for g in GEN:
        print(f"  {g:10s}: pitch_mu={feats[y==g,0].mean():.1f} dur_mu={feats[y==g,3].mean():.2f}")

    # WAV synthesis + spectrogram for one representative melody per genre
    for g in GEN:
        seq = data["seq"][np.where(y == g)[0][0]]
        mu.write_wav(os.path.join(DATA, f"sample_{g}.wav"), seq)
        spec = mu.spectrogram(seq)
        plt.figure(); plt.imshow(spec[:64].T, aspect="auto", origin="lower", cmap="magma")
        plt.title(f"pf01 spectrogram ({g})"); plt.xlabel("frame"); plt.ylabel("freq bin")
        plt.tight_layout(); plt.savefig(f"{FIG}/pf01_spec_{g}.png"); plt.close()
    print("WAV samples + spectrograms written to part_f/data and part_f/figures.")


if __name__ == "__main__":
    main()
