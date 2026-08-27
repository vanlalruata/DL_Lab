"""part_f / music_utils.py

Shared helpers for the music practicals. Provides:
  * a synthetic multi-genre MIDI-style melody dataset (classical / jazz / rock / pop)
    when no real .mid files are present, with an optional real-MIDI loader via
    `pretty_midi` (if you download a dataset such as MAESTRO / Lakh / Wikifonia and
    point download_or_load() at it).
  * feature extraction from note sequences (pitch/duration/velocity statistics),
  * event-tokenization for language-model based composition,
  * WAV synthesis + spectrogram (pure stdlib `wave` + numpy, no extra deps),
  * MIDI export via `pretty_midi` when available, else a .txt/.npz fallback.

Everything degrades gracefully so the practicals run even without pretty_midi/librosa.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GENRES = ["classical", "jazz", "rock", "pop"]


# ----------------------------------------------------------------------------- synthesis
def _genre_params(genre):
    p = {
        "classical": dict(pmin=55, pmax=84, dmean=1.4, dstd=0.3, vel=70, density=0.6, chroma=0.2),
        "jazz":      dict(pmin=50, pmax=82, dmean=0.9, dstd=0.4, vel=80, density=0.8, chroma=0.5),
        "rock":      dict(pmin=38, pmax=67, dmean=0.5, dstd=0.15, vel=95, density=1.0, chroma=0.1),
        "pop":       dict(pmin=52, pmax=79, dmean=0.7, dstd=0.2, vel=85, density=0.9, chroma=0.3),
    }
    return p[genre]


def synthesize_sequence(genre, rng, n_notes=48):
    pr = _genre_params(genre)
    seq = []
    pitch = (pr["pmin"] + pr["pmax"]) // 2
    for _ in range(n_notes):
        # random walk within range; chromatic moves more likely for jazz
        step = rng.choice([-2, -1, 0, 1, 2], p=[0.15, 0.35, 0.05, 0.30, 0.15])
        pitch = int(np.clip(pitch + step, pr["pmin"], pr["pmax"]))
        dur = max(0.25, rng.normal(pr["dmean"], pr["dstd"]))
        vel = int(np.clip(rng.normal(pr["vel"], 8), 40, 127))
        seq.append((pitch, round(dur, 2), vel))
    return seq


def build_dataset(n_per_genre=40, seed=0):
    rng = np.random.RandomState(seed)
    data = {"genre": [], "seq": []}
    for g in GENRES:
        for _ in range(n_per_genre):
            data["genre"].append(g)
            data["seq"].append(synthesize_sequence(g, rng))
    return data


# ----------------------------------------------------------------------------- real MIDI
def load_midi_dir(midi_dir):
    """Load .mid files from `midi_dir/<genre>/*.mid` using pretty_midi if available."""
    try:
        import pretty_midi
    except Exception:
        print("[music_utils] pretty_midi not installed; using synthetic dataset.")
        return None
    data = {"genre": [], "seq": []}
    for g in GENRES:
        d = os.path.join(midi_dir, g)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.lower().endswith(".mid"):
                mid = pretty_midi.PrettyMIDI(os.path.join(d, f))
                seq = []
                for inst in mid.instruments:
                    for n in inst.notes:
                        dur = n.end - n.start
                        seq.append((n.pitch, round(dur, 2), int(n.velocity)))
                if seq:
                    data["genre"].append(g); data["seq"].append(seq)
    return data if data["genre"] else None


def download_or_load(midi_dir=None):
    if midi_dir and os.path.isdir(midi_dir):
        real = load_midi_dir(midi_dir)
        if real:
            print(f"[music_utils] loaded real MIDI from {midi_dir}")
            return real
    print("[music_utils] using synthetic multi-genre dataset")
    return build_dataset()


# ----------------------------------------------------------------------------- features
def extract_features(seq):
    pitches = np.array([n[0] for n in seq], float)
    durs = np.array([n[1] for n in seq], float)
    vels = np.array([n[2] for n in seq], float)
    if len(pitches) == 0:
        return np.zeros(8)
    return np.array([
        pitches.mean(), pitches.std(), pitches.max() - pitches.min(),
        durs.mean(), durs.std(), vels.mean(),
        len(np.unique(pitches)) / len(pitches),            # tonal diversity
        np.diff(pitches).std(),                           # step-size regularity
    ])


# ----------------------------------------------------------------------------- tokenization
def build_vocab(sequences):
    toks = set()
    for seq in sequences:
        for p, d, v in seq:
            toks.add(f"P{p}"); toks.add(f"D{int(round(d*4))}")  # quantize duration
    tok = ["<pad>", "<sos>", "<eos>"] + sorted(toks)
    return {t: i for i, t in enumerate(tok)}, {i: t for i, t in enumerate(tok)}


def tokenize_seq(seq, vocab):
    ids = [vocab["<sos>"]]
    for p, d, v in seq:
        ids.append(vocab[f"P{p}"]); ids.append(vocab[f"D{int(round(d*4))}"])
    ids.append(vocab["<eos>"])
    return ids


def detokenize(ids, ivocab):
    notes = []
    cur_p = None
    for i in ids:
        t = ivocab[i]
        if t in ("<pad>", "<sos>", "<eos>"):
            if t == "<eos>":
                break
            continue
        if t.startswith("P"):
            cur_p = int(t[1:])
        elif t.startswith("D") and cur_p is not None:
            notes.append((cur_p, int(t[1:]) / 4.0, 80))
    return notes


# ----------------------------------------------------------------------------- WAV
def write_wav(path, seq, fs=22050, bpm=120):
    """Sonify a note sequence as a simple additive WAV (stdlib wave + numpy)."""
    import wave
    beat = 60.0 / bpm
    total = int(sum(n[1] for n in seq) * beat * fs) + fs
    buf = np.zeros(max(1, total))
    t0 = 0
    for pitch, dur, vel in seq:
        n = int(dur * beat * fs)
        if n <= 0:
            continue
        tt = np.arange(n) / fs
        f = 440.0 * 2 ** ((pitch - 69) / 12.0)
        sig = np.sin(2 * np.pi * f * tt) * (vel / 127.0)
        # simple decay envelope
        sig *= np.linspace(1.0, 0.3, n)
        buf[t0:t0 + n] += sig
        t0 += n
    buf = np.clip(buf, -1, 1)
    pcm = (buf * 32767).astype("<i2")
    with wave.open(path, "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(fs)
        w.writeframes(pcm.tobytes())


def spectrogram(seq, fs=22050, bpm=120, n_fft=1024):
    beat = 60.0 / bpm
    total = int(sum(n[1] for n in seq) * beat * fs) + fs
    buf = np.zeros(max(1, total))
    t0 = 0
    for pitch, dur, vel in seq:
        n = int(dur * beat * fs)
        if n <= 0:
            continue
        tt = np.arange(n) / fs
        f = 440.0 * 2 ** ((pitch - 69) / 12.0)
        buf[t0:t0 + n] += np.sin(2 * np.pi * f * tt) * (vel / 127.0)
        t0 += n
    win = np.hanning(n_fft)
    hops = max(1, len(buf) // n_fft)
    specs = []
    for s in range(0, len(buf) - n_fft, n_fft):
        seg = buf[s:s + n_fft] * win
        specs.append(np.abs(np.fft.rfft(seg)) ** 2)
    return np.array(specs)


# ----------------------------------------------------------------------------- MIDI export
def save_midi(seq, path):
    try:
        import pretty_midi
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(0, name="synth")
        t = 0.0
        for pitch, dur, vel in seq:
            inst.notes.append(pretty_midi.Note(int(vel), int(pitch),
                                             t, t + float(dur)))
            t += float(dur)
        pm.instruments.append(inst)
        pm.write(path)
        return True
    except Exception:
        # fallback: store as npz
        np.savez(path.replace(".mid", ".npz"),
                 pitches=np.array([n[0] for n in seq]),
                 durs=np.array([n[1] for n in seq]),
                 vels=np.array([n[2] for n in seq]))
        return False


def compose_from_features(target, n_notes=48, seed=0):
    """Generate a novel melody whose statistics match `target` feature vector."""
    rng = np.random.RandomState(seed)
    pmean, pstd, prange, dmean, dstd, _, _, _ = target
    pmin, pmax = max(0, int(pmean - prange / 2)), min(127, int(pmean + prange / 2))
    pitch = int(pmean)
    seq = []
    for _ in range(n_notes):
        pitch = int(np.clip(pitch + rng.normal(0, max(1, pstd)), pmin, pmax))
        dur = max(0.25, rng.normal(dmean, max(0.05, dstd)))
        seq.append((pitch, round(dur, 2), 80))
    return seq
