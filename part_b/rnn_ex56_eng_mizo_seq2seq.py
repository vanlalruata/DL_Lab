"""
Part B — Question 56: English-Mizo T2T Translation using RNN (Seq2Seq with Attention)
Demonstrates: Many-to-Many Seq2Seq with Bahdanau Attention for Machine Translation
- Uses nn.RNN (vanilla RNN) with attention
- POS-aware English encoder for better understanding
- Real parallel corpus: English <-> Mizo (engmiz.txt)
- Best possible translation via attention-weighted decoding

Architecture:
  English Input -> [Word Embedding + POS Embedding] -> Bi-RNN Encoder -> Hidden States
  Mizo Target  -> [Word Embedding] -> RNN Decoder + Attention Context -> Mizo tokens
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# =============================================================================
# 1. DATA LOADING — Parse engmiz.txt parallel corpus
# =============================================================================

def load_parallel_corpus(filepath):
    """Parse engmiz.txt into English-Mizo parallel pairs."""
    word_pairs = []
    sentence_pairs = []

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if '\t' not in line:
                continue
            parts = line.split('\t', 1)
            if len(parts) != 2:
                continue
            eng, mizo = parts[0].strip(), parts[1].strip()
            if not eng or not mizo:
                continue
            if ' ' in eng or ' ' in mizo:
                sentence_pairs.append((eng, mizo))
            else:
                word_pairs.append((eng, mizo))

    return word_pairs, sentence_pairs


def build_training_pairs(word_pairs, sentence_pairs):
    """Combine word dictionary with sentence pairs."""
    all_pairs = sentence_pairs.copy()
    for eng, mizo in word_pairs:
        if eng.lower() not in [e.lower() for e, _ in all_pairs]:
            all_pairs.append((eng, mizo))
    return all_pairs


# =============================================================================
# 2. POS TAGGING — Rule-based POS tagger for English
# =============================================================================

POS_TAGS = ['NOUN', 'VERB', 'ADJ', 'ADV', 'DET', 'PREP', 'PRON', 'AUX', 'NUM', 'OTHER']
POS_PAD = 'OTHER'

NOUN_ENDINGS = ('tion', 'sion', 'ness', 'ment', 'ity', 'er', 'or', 'ism', 'ist', 'ance', 'ence')
VERB_ENDINGS = ('ing', 'ed', 'ize', 'ise', 'ify', 'ate', 'ought', 'ild')
ADJ_ENDINGS = ('able', 'ible', 'al', 'ful', 'ive', 'less', 'ous', 'ish', 'ic')
ADV_ENDINGS = ('ly',)
DET_WORDS = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your',
             'his', 'her', 'its', 'our', 'their', 'i', 'me', 'we', 'they', 'you'}
PREP_WORDS = {'in', 'on', 'at', 'to', 'from', 'by', 'with', 'for', 'of', 'about',
              'into', 'through', 'after', 'before', 'between', 'without', 'within'}
PRON_WORDS = {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
              'us', 'them', 'my', 'your', 'his', 'our', 'their', 'mine', 'yours'}
AUX_WORDS = {'is', 'am', 'are', 'was', 'were', 'been', 'being', 'have', 'has',
              'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should',
              'can', 'could', 'may', 'might', 'must'}


def get_pos(word):
    """Rule-based POS tagger."""
    w = word.lower().strip('.,!?;:\'"')
    if w in DET_WORDS:
        return 'DET'
    if w in PREP_WORDS:
        return 'PREP'
    if w in PRON_WORDS:
        return 'PRON'
    if w in AUX_WORDS:
        return 'AUX'
    if w.isdigit() or any(c.isdigit() for c in w):
        return 'NUM'
    if w.endswith('ly') and len(w) > 3:
        return 'ADV'
    if any(w.endswith(e) for e in ADJ_ENDINGS) and len(w) > 4:
        return 'ADJ'
    if any(w.endswith(e) for e in VERB_ENDINGS) and len(w) > 3:
        return 'VERB'
    if any(w.endswith(e) for e in NOUN_ENDINGS) and len(w) > 3:
        return 'NOUN'
    return 'NOUN'


# =============================================================================
# 3. VOCABULARY
# =============================================================================

class Vocabulary:
    """Vocabulary with special tokens."""

    def __init__(self, name):
        self.name = name
        self.word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.n_words = 4

    def add_sentence(self, sentence):
        for word in sentence.lower().split():
            word = word.strip(".,!?;:\"'()")
            if word and word not in self.word2idx:
                self.word2idx[word] = self.n_words
                self.idx2word[self.n_words] = word
                self.n_words += 1

    def encode(self, sentence):
        return [self.word2idx.get(w.lower().strip(".,!?;:\"'()"), 3)
                for w in sentence.split()]

    def decode(self, indices):
        return [self.idx2word.get(i, '<UNK>') for i in indices]


class POSVocabulary:
    """POS tag vocabulary."""

    def __init__(self):
        self.tag2idx = {POS_PAD: 0}
        self.idx2tag = {0: POS_PAD}
        self.n_tags = 1
        for tag in POS_TAGS:
            if tag not in self.tag2idx:
                self.tag2idx[tag] = self.n_tags
                self.idx2tag[self.n_tags] = tag
                self.n_tags += 1


# =============================================================================
# 4. DATASET
# =============================================================================

class TranslationDataset(torch.utils.data.Dataset):
    """Dataset for English-Mizo translation."""

    def __init__(self, pairs, src_vocab, tgt_vocab, pos_vocab, max_len=12):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.pos_vocab = pos_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        eng, mizo = self.pairs[idx]

        # Source: English with POS
        src_tokens = self.src_vocab.encode(eng)[:self.max_len]
        src_pos = [self.pos_vocab.tag2idx.get(get_pos(w), 0)
                   for w in eng.lower().split()[:self.max_len]]
        src_len = max(len(src_tokens), 1)
        src_tokens = src_tokens + [0] * (self.max_len - len(src_tokens))
        src_pos = src_pos + [0] * (self.max_len - len(src_pos))

        # Target: Mizo with SOS/EOS
        tgt_toks = self.tgt_vocab.encode(mizo)[:self.max_len]
        tgt_input = [1] + tgt_toks[:self.max_len]
        tgt_output = tgt_toks[:self.max_len] + [2]
        tgt_input = tgt_input + [0] * (self.max_len + 1 - len(tgt_input))
        tgt_output = tgt_output + [0] * (self.max_len + 1 - len(tgt_output))

        return (torch.tensor(src_tokens, dtype=torch.long),
                torch.tensor(src_pos, dtype=torch.long),
                torch.tensor(tgt_input, dtype=torch.long),
                torch.tensor(tgt_output, dtype=torch.long))


# =============================================================================
# 5. MODEL — Seq2Seq with Attention + POS embeddings
# =============================================================================

class Encoder(nn.Module):
    """Bi-directional RNN encoder with POS-aware word embeddings."""

    def __init__(self, src_vocab_size, pos_vocab_size, embed_dim, hidden_dim, n_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.word_embedding = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(pos_vocab_size, 8, padding_idx=0)
        combined_dim = embed_dim + 8
        self.rnn = nn.RNN(
            combined_dim, hidden_dim, n_layers,
            batch_first=True, bidirectional=True,
            dropout=0.1 if n_layers > 1 else 0
        )

    def forward(self, src_tokens, src_pos):
        word_emb = self.word_embedding(src_tokens)
        pos_emb = self.pos_embedding(src_pos)
        combined = torch.cat([word_emb, pos_emb], dim=2)
        outputs, hidden = self.rnn(combined)
        return outputs, hidden


class BahdanauAttention(nn.Module):
    """Bahdanau (additive) attention over encoder states."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.Wa = nn.Linear(hidden_dim * 3, hidden_dim)
        self.va = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        batch_size, src_len, _ = encoder_outputs.size()
        dec_hidden = decoder_hidden.unsqueeze(1).expand(batch_size, src_len, -1)
        energy = torch.tanh(self.Wa(torch.cat([dec_hidden, encoder_outputs], dim=2)))
        scores = self.va(energy).squeeze(2)
        weights = F.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights


class Decoder(nn.Module):
    """RNN decoder with attention (no POS needed for generated tokens)."""

    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim, n_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.word_embedding = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        rnn_input_dim = embed_dim + hidden_dim * 2
        self.rnn = nn.RNN(rnn_input_dim, hidden_dim, n_layers, batch_first=True,
                          dropout=0.1 if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)
        self.attention = BahdanauAttention(hidden_dim)

    def forward(self, tgt_tokens, decoder_hidden, encoder_outputs):
        batch_size, tgt_len = tgt_tokens.size()
        word_emb = self.word_embedding(tgt_tokens)
        outputs = []
        all_weights = []
        h = decoder_hidden[-1]

        for t in range(tgt_len):
            context, attn_weights = self.attention(h, encoder_outputs)
            all_weights.append(attn_weights)
            x = torch.cat([word_emb[:, t, :], context], dim=1).unsqueeze(1)
            out, h = self.rnn(x, h.unsqueeze(0))
            out = self.fc(out.squeeze(1))
            outputs.append(out)

        outputs = torch.stack(outputs, dim=1)
        all_weights = torch.stack(all_weights, dim=1)
        return outputs, all_weights


class Seq2SeqRNN(nn.Module):
    """Complete Seq2Seq model: Bi-RNN Encoder + Attention + RNN Decoder."""

    def __init__(self, src_vocab_size, tgt_vocab_size, pos_vocab_size,
                 embed_dim=32, hidden_dim=64, n_layers=1):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, pos_vocab_size, embed_dim, hidden_dim, n_layers)
        self.decoder = Decoder(tgt_vocab_size, embed_dim, hidden_dim, n_layers)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def forward(self, src_tokens, src_pos, tgt_tokens):
        encoder_outputs, encoder_hidden = self.encoder(src_tokens, src_pos)
        if self.n_layers > 1:
            encoder_hidden = encoder_hidden.view(self.n_layers, 2, -1, self.hidden_dim)
            encoder_hidden = encoder_hidden.sum(dim=1)
        else:
            encoder_hidden = encoder_hidden.sum(dim=0, keepdim=True)
        outputs, attention_weights = self.decoder(tgt_tokens, encoder_hidden, encoder_outputs)
        return outputs, attention_weights


# =============================================================================
# 6. TRAINING
# =============================================================================

def train_epoch(model, dataset, optimizer, criterion, device, batch_size=8):
    """Train for one epoch."""
    model.train()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_loss = 0
    total_tokens = 0

    for src_tokens, src_pos, tgt_input, tgt_output in dataloader:
        src_tokens = src_tokens.to(device)
        src_pos = src_pos.to(device)
        tgt_input = tgt_input.to(device)
        tgt_output = tgt_output.to(device)

        optimizer.zero_grad()
        outputs, _ = model(src_tokens, src_pos, tgt_input)
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt_output.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * (tgt_output != 0).sum().item()
        total_tokens += (tgt_output != 0).sum().item()

    return total_loss / max(total_tokens, 1)


def translate(model, sentence, src_vocab, tgt_vocab, pos_vocab, device, max_len=20):
    """Translate English to Mizo using trained model."""
    model.eval()
    with torch.no_grad():
        tokens = src_vocab.encode(sentence)[:max_len]
        pos_tags = [pos_vocab.tag2idx.get(get_pos(w), 0) for w in sentence.lower().split()[:max_len]]
        tokens = tokens + [0] * (max_len - len(tokens))
        pos_tags = pos_tags + [0] * (max_len - len(pos_tags))

        src_t = torch.tensor([tokens], dtype=torch.long).to(device)
        src_p = torch.tensor([pos_tags], dtype=torch.long).to(device)

        encoder_outputs, encoder_hidden = model.encoder(src_t, src_p)

        if model.n_layers > 1:
            encoder_hidden = encoder_hidden.view(model.n_layers, 2, -1, model.hidden_dim)
            encoder_hidden = encoder_hidden.sum(dim=1)
        else:
            encoder_hidden = encoder_hidden.sum(dim=0, keepdim=True)

        decoded = []
        h = encoder_hidden[-1]
        attn_list = []

        for _ in range(max_len):
            context, attn = model.decoder.attention(h, encoder_outputs)
            attn_list.append(attn.cpu().numpy())

            next_token = decoded[-1] if decoded else 1  # <SOS>
            word_emb = model.decoder.word_embedding(torch.tensor([[next_token]], device=device))
            x = torch.cat([word_emb.squeeze(1), context], dim=1).unsqueeze(1)
            out, h = model.decoder.rnn(x, h.unsqueeze(0))
            out = model.decoder.fc(out.squeeze(1))
            next_token = out.argmax(dim=1).item()

            if next_token == 2:  # <EOS>
                break
            decoded.append(next_token)

        mizo_tokens = tgt_vocab.decode(decoded)
        return ' '.join(mizo_tokens), attn_list


# =============================================================================
# 7. MAIN
# =============================================================================

def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    corpus_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'engmiz.txt')
    word_pairs, sentence_pairs = load_parallel_corpus(corpus_path)

    print(f"\nLoaded corpus:")
    print(f"  Word-level pairs: {len(word_pairs)}")
    print(f"  Sentence-level pairs: {len(sentence_pairs)}")

    all_pairs = build_training_pairs(word_pairs, sentence_pairs)
    print(f"  Total training pairs: {len(all_pairs)}")

    print("\nSample parallel pairs:")
    for i in range(min(5, len(all_pairs))):
        eng, mizo = all_pairs[i]
        print(f"  EN: '{eng}' -> MI: '{mizo}'")

    # Build vocabularies
    src_vocab = Vocabulary('English')
    tgt_vocab = Vocabulary('Mizo')
    pos_vocab = POSVocabulary()

    for eng, mizo in all_pairs:
        src_vocab.add_sentence(eng)
        tgt_vocab.add_sentence(mizo)

    print(f"\nVocabulary size:")
    print(f"  English: {src_vocab.n_words}")
    print(f"  Mizo: {tgt_vocab.n_words}")
    print(f"  POS tags: {pos_vocab.n_tags} ({POS_TAGS})")

    # Create dataset and model
    max_seq_len = 12
    dataset = TranslationDataset(all_pairs, src_vocab, tgt_vocab, pos_vocab, max_seq_len)

    embed_dim = 32
    hidden_dim = 64
    model = Seq2SeqRNN(
        src_vocab_size=src_vocab.n_words,
        tgt_vocab_size=tgt_vocab.n_words,
        pos_vocab_size=pos_vocab.n_tags,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_layers=1
    ).to(device)

    print(f"\nModel: Seq2SeqRNN with Bahdanau Attention + POS Encoder")
    print(f"  Encoder: Bi-RNN({embed_dim}+8 -> {hidden_dim}x2) + POS embedding")
    print(f"  Decoder: RNN({embed_dim}+{hidden_dim}x2 -> {hidden_dim}) + Attention")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 30
    batch_size = 8

    print(f"\nTraining for {num_epochs} epochs...")
    print("-" * 50)

    for epoch in range(num_epochs):
        loss = train_epoch(model, dataset, optimizer, criterion, device, batch_size)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss:.4f}")

    print("-" * 50)
    print("Training complete!")

    # Demonstrate translations
    print("\n" + "=" * 60)
    print("ENGLISH -> MIZO TRANSLATION (Seq2Seq RNN with Attention)")
    print("=" * 60)

    test_sentences = [
        "I am fine",
        "What is your name",
        "How are you",
        "I love you",
        "Thank you",
        "Where are you going",
        "I am feeling lonely",
        "This is my first time in Mizoram",
        "God is good to me",
        "I will teach you",
        "Hello John are you okay",
        "Nice to hear",
        "All is well",
        "Get well soon",
        "I need a doctor",
    ]

    print("\nTranslation Results:")
    print("-" * 50)
    for sent in test_sentences:
        mizo_translation, attn = translate(model, sent, src_vocab, tgt_vocab, pos_vocab, device)
        pos_info = " | ".join([f"{w}/{get_pos(w)}" for w in sent.lower().split()])
        print(f"EN: '{sent}'")
        print(f"  POS: {pos_info}")
        print(f"MI: '{mizo_translation}'")
        print()

    # Attention analysis
    print("=" * 60)
    print("ATTENTION ANALYSIS")
    print("=" * 60)
    sent = "I am fine"
    mizo_trans, attn = translate(model, sent, src_vocab, tgt_vocab, pos_vocab, device)
    tokens = sent.lower().split()
    print(f"Source: {sent}")
    print(f"Translation: {mizo_trans}")
    if attn and len(attn) > 0:
        print("\nAttention weights (first 3 decoder steps):")
        for t, w in enumerate(attn[:3]):
            weights = w.flatten() if hasattr(w, 'flatten') else w
            print(f"  Step {t}:")
            for i, tok in enumerate(tokens):
                if i < len(weights):
                    bar = "█" * int(weights[i] * 20)
                    print(f"    {tok}: {weights[i]:.3f} {bar}")
    print()

    # POS insights
    print("=" * 60)
    print("POS-AWARE TRANSLATION INSIGHTS")
    print("=" * 60)
    print("""
    The encoder uses POS embeddings as additional input:
    - English word + POS embedding (32+8 dimensions) -> Bi-RNN encoder
    - POS helps encoder distinguish word roles:
      * "the cat sat" vs "the sat cat" (DET+NOUN vs DET+VERB+NOUN)
      * POS tags guide attention to semantically important words
    
    Mizo POS patterns from corpus:
    - NOUN: chaw (food), in (home), tui (water), pumpui (stomach)
    - VERB: ei (take), kal (go), dam (be/fine), thei (eat)
    - DET: ka (the), mi (I/me), nang (you)
    - ADJ: hle mai (beautiful), lawm (good/okay)
    - ADV: rawh (quickly), tur (forward)
    
    Best translation achieved via:
    1. Bahdanau attention aligns source and target words
    2. POS context helps resolve ambiguity
    3. Greedy decoding selects highest probability tokens
    """)

    # Save model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mizen_trans_model.pth')
    torch.save({
        'model_state': model.state_dict(),
        'src_vocab': src_vocab,
        'tgt_vocab': tgt_vocab,
        'pos_vocab': pos_vocab,
    }, model_path)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
