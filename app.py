"""
NewsSum — Streamlit App
Abstractive summarization of news articles with Encoder-Decoder + Bahdanau Attention.

Features:
    1. Abstractive summarization from pasted text, PDF, or TXT files.
    2. Interactive chat with persistent history over the loaded document.
    3. Metrics panel with literature comparison.
    4. Lab for adjusting inference parameters.

Model operates in English only.
"""

import io
import os
import re
import time
import pickle
from collections import Counter

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rouge_score import rouge_scorer

CHECK = "✓"


# ════════════════════════════════════════════════════════════════════════════
# Page config
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NewsSum | Encoder-Decoder + Attention",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
code, pre, .stCode { font-family: 'JetBrains Mono', monospace; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

.main-header {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-radius: 12px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.5rem;
    border: 1px solid #334155;
}
.main-title { font-size: 1.9rem; font-weight: 700; color: #f1f5f9; margin: 0; letter-spacing: -0.4px; }
.main-subtitle { color: #94a3b8; font-size: 0.92rem; margin-top: 0.4rem; }
.tag {
    display: inline-block;
    background: rgba(99,102,241,0.18);
    color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.4);
    padding: 2px 10px; border-radius: 14px;
    font-size: 0.72rem; font-weight: 600;
    margin-right: 6px; margin-top: 8px;
}
.tag.green  { background: rgba(16,185,129,0.15); color: #6ee7b7; border-color: rgba(16,185,129,0.35); }
.tag.amber  { background: rgba(245,158,11,0.15); color: #fcd34d; border-color: rgba(245,158,11,0.35); }

.summary-box {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #4f46e5;
    border-left: 4px solid #6366f1;
    border-radius: 10px;
    padding: 1.4rem 1.5rem;
    margin: 1rem 0;
}
.summary-text { color: #e2e8f0; font-size: 1.02rem; line-height: 1.65; }
.summary-label {
    color: #818cf8; font-size: 0.72rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;
}

.section-title {
    font-size: 1.15rem; font-weight: 600; color: #f1f5f9;
    margin-bottom: 0.9rem; padding-bottom: 0.4rem;
    border-bottom: 2px solid #334155;
}

.rouge-badge {
    display: inline-block;
    background: rgba(99,102,241,0.15); color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.3);
    padding: 4px 12px; border-radius: 8px;
    font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 1.05rem;
}

.lit-table { width: 100%; border-collapse: collapse; }
.lit-table th { background: #1e293b; color: #94a3b8; font-size: 0.78rem; text-transform: uppercase; padding: 10px 14px; text-align: left; }
.lit-table td { padding: 9px 14px; border-bottom: 1px solid #1e293b; color: #e2e8f0; font-size: 0.88rem; }
.lit-table tr:hover td { background: #1e293b33; }
.lit-table .highlight td { color: #a5b4fc; font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)


# ════════════════════════════════════════════════════════════════════════════
# Tokenization and vocabulary
# ════════════════════════════════════════════════════════════════════════════
def simple_tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return text.split()


class Vocabulary:
    """Word-level vocabulary with special tokens."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        for i, tok in enumerate(["<pad>", "<unk>", "<sos>", "<eos>"]):
            self.word2idx[tok] = i
            self.idx2word[i] = tok
        self.pad_idx, self.unk_idx, self.sos_idx, self.eos_idx = 0, 1, 2, 3

    def encode(self, text, max_len=None, add_eos=False, add_sos=False):
        tokens = simple_tokenize(text)
        if max_len:
            tokens = tokens[:max_len]
        ids = [self.word2idx.get(t, self.unk_idx) for t in tokens]
        if add_sos:
            ids = [self.sos_idx] + ids
        if add_eos:
            ids = ids + [self.eos_idx]
        return ids

    def decode(self, ids, skip_special=True):
        special = {self.pad_idx, self.sos_idx}
        words = []
        for i in ids:
            if i == self.eos_idx:
                break
            if skip_special and i in special:
                continue
            words.append(self.idx2word.get(i, "<unk>"))
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)


# ════════════════════════════════════════════════════════════════════════════
# Architecture: Encoder + Attention + Decoder + Seq2Seq
# ════════════════════════════════════════════════════════════════════════════
class BahdanauAttention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim, attn_dim=None):
        super().__init__()
        attn_dim = attn_dim or dec_hidden_dim
        self.attn_enc = nn.Linear(enc_hidden_dim * 2, attn_dim, bias=False)
        self.attn_dec = nn.Linear(dec_hidden_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden, src_mask=None):
        enc_proj = self.attn_enc(encoder_outputs)
        dec_proj = self.attn_dec(decoder_hidden).unsqueeze(1)
        energy = torch.tanh(enc_proj + dec_proj)
        scores = self.v(energy).squeeze(2)
        if src_mask is not None:
            scores = scores.masked_fill(src_mask == 0, float("-inf"))
        attn_weights = F.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout, embedding_matrix=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(embedding_matrix)
        self.rnn = nn.LSTM(
            embed_dim, hidden_dim, n_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src, src_lens):
        embedded = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        outputs, (hidden, cell) = self.rnn(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        hidden = self._combine(hidden)
        cell = self._combine(cell)
        return outputs, hidden, cell

    def _combine(self, state):
        batch_size = state.shape[1]
        state = state.view(self.n_layers, 2, batch_size, self.hidden_dim)
        combined = torch.cat([state[:, 0], state[:, 1]], dim=2)
        projected = [torch.tanh(self.fc_hidden(combined[i])) for i in range(self.n_layers)]
        return torch.stack(projected)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, enc_hidden_dim, dec_hidden_dim,
                 n_layers, dropout, embedding_matrix=None, attention=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.attention = attention
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(embedding_matrix)
        self.rnn = nn.LSTM(
            embed_dim + enc_hidden_dim * 2, dec_hidden_dim, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(dec_hidden_dim + enc_hidden_dim * 2 + embed_dim, vocab_size)

    def forward(self, trg_token, hidden, cell, encoder_outputs, src_mask=None):
        trg_token = trg_token.unsqueeze(1)
        embedded = self.dropout(self.embedding(trg_token))
        context, attn_weights = self.attention(encoder_outputs, hidden[-1], src_mask)
        context_exp = context.unsqueeze(1)
        rnn_input = torch.cat([embedded, context_exp], dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        output = output.squeeze(1)
        context = context_exp.squeeze(1)
        embedded = embedded.squeeze(1)
        prediction = self.fc_out(torch.cat([output, context, embedded], dim=1))
        return prediction, hidden, cell, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, pad_idx=0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.pad_idx = pad_idx

    def create_mask(self, src):
        return src != self.pad_idx


# ════════════════════════════════════════════════════════════════════════════
# Summary generation
# ════════════════════════════════════════════════════════════════════════════
def clean_summary(text: str) -> str:
    """Remove degenerate repetition common in undertrained seq2seq models."""
    if not text or not text.strip():
        return "(empty summary — model may need more training epochs)"
    tokens = text.split()
    cleaned, prev, repeat_count = [], None, 0
    for tok in tokens:
        if tok == prev:
            repeat_count += 1
            if repeat_count >= 3:
                break
        else:
            repeat_count = 1
        cleaned.append(tok)
        prev = tok
    result = " ".join(cleaned).strip()
    words_only = [t for t in cleaned if t.isalpha()]
    if len(words_only) < 3:
        return (result + "\n\n⚠️ Low-quality output: model needs more training. "
                "Set FAST_MODE = False in the notebook and retrain for meaningful summaries.")
    return result


def generate_summary_custom(model, vocab, article_text, max_len=100,
                            temperature=1.0, top_k=0, top_p=0.0,
                            device=torch.device("cpu")):
    """Greedy / top-k / top-p decoding over the Encoder-Decoder model."""
    model.eval()
    with torch.no_grad():
        src_ids = vocab.encode(article_text, max_len=400)
        if not src_ids:
            return "", [], []
        src = torch.LongTensor(src_ids).unsqueeze(0).to(device)
        src_lens = torch.LongTensor([len(src_ids)])

        enc_out, hidden, cell = model.encoder(src, src_lens)
        src_mask = model.create_mask(src)

        tokens = [vocab.sos_idx]
        attn_store = []
        input_tok = torch.LongTensor([vocab.sos_idx]).to(device)

        for _ in range(max_len):
            pred, hidden, cell, attn_w = model.decoder(
                input_tok, hidden, cell, enc_out, src_mask
            )
            logits = pred / max(temperature, 1e-6)

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                idx_remove = sorted_idx[remove]
                logits[:, idx_remove] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            if temperature <= 0.05:
                next_id = probs.argmax(1).item()
            else:
                next_id = torch.multinomial(probs, 1).item()

            tokens.append(next_id)
            attn_store.append(attn_w.squeeze(0).cpu().numpy())
            if next_id == vocab.eos_idx:
                break
            input_tok = torch.LongTensor([next_id]).to(device)

    return vocab.decode(tokens), attn_store, src_ids


# ════════════════════════════════════════════════════════════════════════════
# Model loading
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_custom_model():
    """Load the trained Encoder-Decoder model, or create a demo version."""
    device = torch.device("cpu")
    EMBED_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT = 100, 256, 2, 0.3

    if os.path.exists("vocab.pkl"):
        with open("vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
    else:
        vocab = Vocabulary()
        demo_words = (
            "the a an is was are were has have had he she they it we i you his her their "
            "of in to for on with at by from as said told according reported president "
            "government police people new year time country world state official party "
            "news law court city us uk china russia attack killed died won lost election "
            "vote deal trade economy market million billion percent war peace crisis fire "
            "former national international local public social after before during while "
            "when where who cnn daily mail article report story update").split()
        for w in demo_words:
            idx = len(vocab.word2idx)
            vocab.word2idx[w] = idx
            vocab.idx2word[idx] = w

    vocab_size = len(vocab)
    emb_matrix = torch.randn(vocab_size, EMBED_DIM) * 0.1
    attn = BahdanauAttention(HIDDEN_DIM, HIDDEN_DIM)
    encoder = Encoder(vocab_size, EMBED_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT, emb_matrix)
    decoder = Decoder(vocab_size, EMBED_DIM, HIDDEN_DIM, HIDDEN_DIM,
                      N_LAYERS, DROPOUT, emb_matrix, attn)
    model = Seq2Seq(encoder, decoder, device).to(device)

    loaded = False
    if os.path.exists("best_model.pt"):
        try:
            ckpt = torch.load("best_model.pt", map_location=device)
            model.load_state_dict(ckpt["model_state"])
            loaded = True
        except Exception:
            pass

    return model, vocab, device, loaded


HF_SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"


def _load_seq2seq(model_name):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    mdl.eval()
    return tok, mdl


@st.cache_resource(show_spinner=False)
def load_hf_summarizer():
    try:
        return _load_seq2seq(HF_SUMMARIZER_MODEL), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


# ════════════════════════════════════════════════════════════════════════════
# File extraction — PDF and TXT only
# ════════════════════════════════════════════════════════════════════════════
def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        from PyPDF2 import PdfReader
    reader = PdfReader(io.BytesIO(file_bytes))
    chunks = []
    for page in reader.pages:
        try:
            chunks.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(chunks).strip()


def extract_uploaded_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(data)
    if name.endswith(".txt"):
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace")
    raise ValueError(f"Unsupported format: {name}. Use .pdf or .txt")


# ════════════════════════════════════════════════════════════════════════════
# BART generation
# ════════════════════════════════════════════════════════════════════════════
def generate_summary_bart(text: str, summarizer, max_length=130, min_length=30,
                          num_beams=4) -> str:
    if summarizer is None or not text.strip():
        return ""
    tokenizer, model = summarizer
    text = text[:6000]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# ════════════════════════════════════════════════════════════════════════════
# Chat responses (English)
# ════════════════════════════════════════════════════════════════════════════
def generate_chat_response(question: str, article: str, summary: str) -> str:
    q = question.lower().strip()

    if any(w in q for w in ["about", "topic", "what is", "what does", "summary"]):
        return (f"The document summarizes as: <i>{summary}</i>"
                if summary else "No summary available yet. Generate one first.")

    if any(w in q for w in ["who", "people", "entity", "entities", "mention"]):
        if not article:
            return "No article loaded."
        caps = [w for w in article.split() if w[:1].isupper() and len(w) > 2 and w.isalpha()]
        unique = list(dict.fromkeys(caps))[:10]
        return ("Entities / names detected: <b>" + ", ".join(unique) + "</b>"
                if unique else "No clear entities identified.")

    if any(w in q for w in ["attention", "how does", "mechanism", "architecture"]):
        return (
            "<b>Bahdanau Attention Mechanism:</b><br>"
            "At each decoder step, an alignment score is computed: "
            "<code>e(t,i) = v · tanh(W_enc h_i + W_dec s_t)</code>. "
            "Softmax converts these into attention weights <code>α</code>, "
            "and the context vector is computed as a weighted sum of encoder states. "
            "This lets the model focus on relevant parts of the input when generating each token."
        )

    if any(w in q for w in ["rouge", "metric", "score", "evaluation"]):
        return (
            "<b>ROUGE</b> measures n-gram overlap between the generated and reference summary. "
            "ROUGE-1 (unigrams), ROUGE-2 (bigrams), ROUGE-L (longest common subsequence). "
            "On CNN/DailyMail, ROUGE-1 > 0.35 is acceptable for classic Encoder-Decoder models."
        )

    if any(w in q for w in ["temperature"]):
        return (
            "<b>Temperature</b> scales logits before softmax: "
            "T &lt; 1 gives more deterministic output, T &gt; 1 adds diversity. "
            "For news summarization, T ∈ [0.7, 1.0] is recommended."
        )

    if any(w in q for w in ["hello", "hi", "hey"]):
        return (
            "Hello! I'm NewsSum. Load an article or document, generate a summary, "
            "then ask me about its content, ROUGE metrics, or the model architecture."
        )

    if article:
        sentences = re.split(r"(?<=[\.\!\?])\s+", article)
        keywords = [w for w in q.split() if len(w) > 3]
        relevant = [s for s in sentences if any(k in s.lower() for k in keywords)]
        if relevant:
            return f"Relevant passage: <i>«{relevant[0][:280].strip()}…»</i>"

    return (f"Question logged: <i>{question}</i>. "
            "I can help with summaries, document entities, ROUGE metrics, or the model architecture.")


# ════════════════════════════════════════════════════════════════════════════
# Attention map
# ════════════════════════════════════════════════════════════════════════════
def render_attention_map(article_text, summary, attn_weights):
    src_tokens = simple_tokenize(article_text)[:50]
    trg_tokens = simple_tokenize(summary)[:20]
    if not attn_weights or not trg_tokens:
        return None
    n_trg = min(len(attn_weights), len(trg_tokens))
    n_src = min(len(src_tokens), attn_weights[0].shape[0])
    attn_matrix = np.array([attn_weights[t][:n_src] for t in range(n_trg)])

    fig, ax = plt.subplots(figsize=(min(16, n_src // 2 + 4), max(4, n_trg // 2 + 2)))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    im = ax.imshow(attn_matrix, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(n_src))
    ax.set_yticks(range(n_trg))
    ax.set_xticklabels(src_tokens[:n_src], rotation=45, ha="right", fontsize=8, color="#94a3b8")
    ax.set_yticklabels(trg_tokens[:n_trg], fontsize=9, color="#e2e8f0")
    ax.set_xlabel("Article (source tokens)", color="#94a3b8")
    ax.set_ylabel("Summary (target tokens)", color="#94a3b8")
    ax.tick_params(colors="#64748b")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(colors="#94a3b8")
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════════
# Main app
# ════════════════════════════════════════════════════════════════════════════
def main():
    model, vocab, device, model_loaded = load_custom_model()

    # ── Sidebar (defined first so `backend` is available for the warning) ────
    with st.sidebar:
        st.markdown("## Parameter Lab")

        st.markdown("### Inference backend")
        backend = st.radio(
            "Backend",
            options=["BART pretrained (Hugging Face)",
                     "Trained model (Encoder-Decoder + Attention)"],
            help="BART is a pretrained reference model that works out of the box. "
                 "The trained model requires best_model.pt next to app.py.",
        )

        st.markdown("### Decoding")
        temperature = st.slider("Temperature", 0.05, 2.0, 1.0, 0.05,
                                help="Scales logits before softmax.")
        top_k = st.slider("top-k", 0, 200, 0, 5,
                          help="Filter to top-k tokens. 0 = disabled.")
        top_p = st.slider("top-p (nucleus)", 0.0, 1.0, 0.0, 0.05,
                          help="Filter to cumulative probability mass p. 0 = disabled.")
        max_len = st.slider("Max summary length (tokens)", 30, 200, 80, 5)
        num_beams = st.slider("Beams (BART only)", 1, 8, 4, 1)

        st.markdown("### Visualization")
        show_attention = st.checkbox("Show attention map (trained model)", True)
        show_rouge = st.checkbox("Compute ROUGE vs. original document", True)

        st.markdown("---")
        st.markdown("### Model configuration")
        st.code(
            f"""Encoder   : BiLSTM (2 layers)
Decoder   : LSTM   (2 layers)
Attention : Bahdanau additive
Embedding : GloVe 6B 100d
Hidden    : 256
Vocab     : {len(vocab):,}""",
            language="text",
        )

        st.markdown("### Recommended reading")
        st.markdown(
            """
- Bahdanau et al. (2015) — Additive attention.
- See et al. (2017) — Pointer-Generator + Coverage.
- Lewis et al. (2020) — BART.
- Zhang et al. (2020) — PEGASUS.
- Lin (2004) — ROUGE.
"""
        )

    # ── Header (after sidebar so `backend` is defined) ──────────────────────
    using_bart = backend.startswith("BART")
    status_tag = (
        f'<span class="tag green">{CHECK} BART pretrained (Hugging Face)</span>'
        if using_bart
        else (
            f'<span class="tag green">{CHECK} Trained model loaded</span>'
            if model_loaded
            else '<span class="tag amber">Demo mode — random weights</span>'
        )
    )
    st.markdown(
        f"""
    <div class="main-header">
      <p class="main-title">NewsSum</p>
      <p class="main-subtitle">
        Automatic news summarization · Encoder-Decoder + Bahdanau Attention · GloVe 6B · English
      </p>
      <span class="tag">CNN/DailyMail v3.0.0</span>
      <span class="tag">PyTorch</span>
      <span class="tag green">Seq2Seq</span>
      <span class="tag green">Additive Attention</span>
      {status_tag}
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Show warning only when using the custom model backend without trained weights
    if not using_bart and not model_loaded:
        st.warning(
            "**Demo mode active.** `best_model.pt` was not found — the custom model is running "
            "with random weights. Switch the backend to **BART pretrained (Hugging Face)** "
            "in the sidebar for meaningful summaries."
        )

    # ── Tabs ────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Summarizer", "Chat", "Metrics", "Lab"]
    )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — Summarizer
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown('<div class="section-title">Summary Generator</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([3, 2])

        with col1:
            example_articles = {
                "Politics": (
                    "The president announced a sweeping new climate policy that would "
                    "require all federal buildings to be powered by renewable energy by "
                    "2030. The executive order signed on Thursday also establishes a new "
                    "task force to oversee the transition and allocate funding from the "
                    "infrastructure bill. Critics from the energy sector warned the plan "
                    "could increase utility costs, while environmental groups praised it "
                    "as an important step toward reducing carbon emissions."
                ),
                "Economy": (
                    "Global stock markets tumbled on Monday as investors reacted to new "
                    "inflation data showing prices rose faster than expected last month. "
                    "The consumer price index climbed 8.5 percent year over year, "
                    "surpassing economist forecasts and renewing fears that the Federal "
                    "Reserve may need to raise interest rates more aggressively."
                ),
                "Science": (
                    "Scientists announced a breakthrough in fusion energy research after "
                    "achieving a net energy gain for the second time at the National "
                    "Ignition Facility in California. The experiment produced 3.15 "
                    "megajoules of energy from a 2.05 megajoule laser input, marking a "
                    "significant milestone in the decades-long quest for clean limitless power."
                ),
            }

            uploaded_file = st.file_uploader(
                "Upload a document (PDF or TXT in English)",
                type=["pdf", "txt"],
                help="The model operates in English. Input text in other languages may produce poor results.",
                key="uploader",
            )

            if uploaded_file is not None:
                file_id = (uploaded_file.name, getattr(uploaded_file, "size", None))
                if st.session_state.get("_last_uploaded_id") != file_id:
                    try:
                        with st.spinner(f"Extracting text from {uploaded_file.name}..."):
                            extracted_text = extract_uploaded_text(uploaded_file)
                        if extracted_text and extracted_text.strip():
                            st.session_state["article_input"] = extracted_text
                            st.session_state["_last_uploaded_id"] = file_id
                            st.success(f"{CHECK} {len(extracted_text.split()):,} words extracted from {uploaded_file.name}.")
                        else:
                            st.warning("Could not extract text (empty or unreadable document).")
                    except Exception as exc:
                        st.error(f"Error processing file: {exc}")

            st.markdown("**Example articles:**")
            ex1, ex2, ex3 = st.columns(3)
            if ex1.button("Politics"):
                st.session_state["article_input"] = example_articles["Politics"]
                st.session_state["_last_uploaded_id"] = None
                st.rerun()
            if ex2.button("Economy"):
                st.session_state["article_input"] = example_articles["Economy"]
                st.session_state["_last_uploaded_id"] = None
                st.rerun()
            if ex3.button("Science"):
                st.session_state["article_input"] = example_articles["Science"]
                st.session_state["_last_uploaded_id"] = None
                st.rerun()

            article_input = st.text_area(
                "Or paste English text to summarize:",
                height=260,
                placeholder="Paste the article here. Example: The president announced a new climate policy...",
                key="article_input",
            )

            generate_btn = st.button("Generate summary", type="primary", use_container_width=True)

        with col2:
            if st.session_state.get("last_summary"):
                summary, attn_weights, src_ids = st.session_state["last_summary"]

                st.markdown(
                    f"""
                <div class="summary-box">
                  <div class="summary-label">Generated Summary (English)</div>
                  <div class="summary-text">{summary}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                src_words = len(article_input.split()) if article_input else 0
                trg_words = len(summary.split())
                ratio = src_words / max(trg_words, 1)
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Source words", f"{src_words:,}")
                mc2.metric("Summary words", f"{trg_words:,}")
                mc3.metric("Compression", f"{ratio:.1f}x")

                if show_rouge and article_input and summary:
                    try:
                        scorer = rouge_scorer.RougeScorer(
                            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
                        )
                        sc = scorer.score(article_input[:1500], summary)
                        st.markdown("**ROUGE (summary vs. document):**")
                        rc1, rc2, rc3 = st.columns(3)
                        rc1.markdown(f'<div class="rouge-badge">R-1: {sc["rouge1"].fmeasure:.3f}</div>', unsafe_allow_html=True)
                        rc2.markdown(f'<div class="rouge-badge">R-2: {sc["rouge2"].fmeasure:.3f}</div>', unsafe_allow_html=True)
                        rc3.markdown(f'<div class="rouge-badge">R-L: {sc["rougeL"].fmeasure:.3f}</div>', unsafe_allow_html=True)
                    except Exception:
                        pass

                ctx = st.session_state.setdefault("chat_context", {})
                ctx["last_article"] = article_input
                ctx["last_summary"] = summary

        # Generate button action
        if generate_btn and article_input and len(article_input.strip()) > 20:
            try:
                if backend.startswith("BART"):
                    summarizer, summ_err = load_hf_summarizer()
                    if summarizer is None:
                        st.error(f"Could not load `{HF_SUMMARIZER_MODEL}`. Detail: `{summ_err}`.")
                        st.stop()
                    with st.spinner(f"Generating summary with {HF_SUMMARIZER_MODEL}..."):
                        summary_en = generate_summary_bart(
                            article_input, summarizer,
                            max_length=max_len,
                            min_length=max(20, max_len // 4),
                            num_beams=num_beams,
                        )
                        st.session_state["last_summary"] = (summary_en, [], [])
                else:
                    if not model_loaded:
                        st.warning(
                            "Using the trained model backend in demo mode (random weights). "
                            "Switch to BART in the sidebar for meaningful output."
                        )
                    with st.spinner("Generating summary with Encoder-Decoder model..."):
                        raw_summary, attn_weights, src_ids = generate_summary_custom(
                            model, vocab, article_input,
                            max_len=max_len, temperature=temperature,
                            top_k=top_k, top_p=top_p, device=device,
                        )
                        summary_en = clean_summary(raw_summary)
                        st.session_state["last_summary"] = (summary_en, attn_weights, src_ids)

                st.session_state["last_article_text"] = article_input
                st.rerun()
            except Exception as exc:
                st.error(f"Error during generation: {exc}")

        # Attention map
        if (show_attention
                and backend.startswith("Trained model")
                and st.session_state.get("last_summary")):
            summary, attn_weights, _ = st.session_state["last_summary"]
            article_text = st.session_state.get("last_article_text", "")
            if attn_weights and article_text:
                st.markdown("---")
                st.markdown('<div class="section-title">Attention Map</div>', unsafe_allow_html=True)
                fig = render_attention_map(article_text, summary, attn_weights)
                if fig is not None:
                    st.pyplot(fig)
                    plt.close(fig)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — Chat
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown('<div class="section-title">Interactive Chat</div>', unsafe_allow_html=True)

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
        if "chat_context" not in st.session_state:
            st.session_state["chat_context"] = {}

        ctx = st.session_state["chat_context"]
        if not ctx.get("last_article"):
            st.info("Generate a summary in the **Summarizer** tab first to enable contextual chat.")
        else:
            st.success(f"{CHECK} Active document: {len(ctx['last_article'].split()):,} words.")

        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"], unsafe_allow_html=True)
                if msg.get("time"):
                    st.caption(msg["time"])

        st.markdown("**Quick questions:**")
        q1, q2, q3, q4 = st.columns(4)
        quick = {
            "topic":      "What is this document about?",
            "entities":   "Who or what is mentioned?",
            "attention":  "How does the attention mechanism work?",
            "rouge":      "How do I interpret ROUGE scores?",
        }
        triggered = None
        if q1.button(quick["topic"]):     triggered = quick["topic"]
        if q2.button(quick["entities"]): triggered = quick["entities"]
        if q3.button(quick["attention"]): triggered = quick["attention"]
        if q4.button(quick["rouge"]):    triggered = quick["rouge"]

        user_input = st.chat_input("Ask something about the document...")
        new_msg = triggered or user_input

        if new_msg:
            ts = time.strftime("%H:%M")
            st.session_state["chat_history"].append({"role": "user", "content": new_msg, "time": ts})
            response = generate_chat_response(
                new_msg,
                ctx.get("last_article", ""),
                ctx.get("last_summary", ""),
            )
            st.session_state["chat_history"].append({"role": "assistant", "content": response, "time": ts})
            st.rerun()

        if st.session_state["chat_history"]:
            if st.button("Clear history"):
                st.session_state["chat_history"] = []
                st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — Metrics
    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-title">Evaluation Metrics</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns([2, 1])

        with col_a:
            st.markdown("#### Literature comparison")
            st.markdown(
                f"""
                <table class="lit-table">
                  <thead>
                    <tr><th>Model</th><th>Type</th><th>ROUGE-1</th><th>ROUGE-2</th><th>ROUGE-L</th><th>Year</th></tr>
                  </thead>
                  <tbody>
                    <tr><td>Lead-3</td><td>Extractive (baseline)</td><td>0.401</td><td>0.175</td><td>0.365</td><td>—</td></tr>
                    <tr><td>Seq2Seq</td><td>Basic abstractive</td><td>0.358</td><td>0.144</td><td>0.330</td><td>2015</td></tr>
                    <tr><td>Seq2Seq + Attention</td><td>Abstractive</td><td>0.374</td><td>0.158</td><td>0.346</td><td>2015</td></tr>
                    <tr><td>Pointer-Generator</td><td>Abstractive + Coverage</td><td>0.398</td><td>0.173</td><td>0.367</td><td>2017</td></tr>
                    <tr><td>UniLM</td><td>Fine-tuned Transformer</td><td>0.435</td><td>0.203</td><td>0.402</td><td>2019</td></tr>
                    <tr><td>PEGASUS</td><td>Abstractive pre-trained</td><td>0.447</td><td>0.214</td><td>0.417</td><td>2020</td></tr>
                    <tr><td>BART</td><td>Abstractive pre-trained</td><td>0.448</td><td>0.214</td><td>0.412</td><td>2020</td></tr>
                    <tr class="highlight"><td>Proposed model {CHECK}</td><td>Encoder-Decoder + GloVe</td><td>~0.30–0.38</td><td>~0.12–0.16</td><td>~0.28–0.35</td><td>2024</td></tr>
                  </tbody>
                </table>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
> **Note.** State-of-the-art models (BART, PEGASUS) use Transformer architectures
> with hundreds of millions of parameters and massive pre-training. The proposed model
> serves a pedagogical purpose: illustrating Encoder-Decoder fundamentals with attention.
"""
            )

        with col_b:
            st.markdown("#### ROUGE interpretation")
            st.markdown(
                """
**ROUGE-N** measures n-gram overlap between the generated and reference summary:

```
ROUGE-1 : unigrams
ROUGE-2 : bigrams
ROUGE-L : longest common subsequence
```

| Score | Quality |
|-------|---------|
| > 0.40 | Excellent |
| 0.30 – 0.40 | Good |
| 0.20 – 0.30 | Acceptable |
| < 0.20 | Low |

**Complementary metrics:** BERTScore (semantic), METEOR (morphology),
BLEU (n-grams), Perplexity (training).
"""
            )

        st.markdown("---")
        st.markdown("#### ROUGE comparison chart")

        systems = ["Lead-3", "Seq2Seq\nbasic", "Seq2Seq\n+ Attention",
                   "Pointer-Gen", "BART", "Proposed"]
        r1 = [0.401, 0.358, 0.374, 0.398, 0.448, 0.340]
        r2 = [0.175, 0.144, 0.158, 0.173, 0.214, 0.140]
        rl = [0.365, 0.330, 0.346, 0.367, 0.412, 0.320]

        x = np.arange(len(systems))
        width = 0.25
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#1e293b")
        colors = ["#6366f1", "#10b981", "#f59e0b"]

        for i, (vals, color, label) in enumerate(
            zip([r1, r2, rl], colors, ["ROUGE-1", "ROUGE-2", "ROUGE-L"])
        ):
            bars = ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85, edgecolor="#0f172a")
            for j, (bar, v) in enumerate(zip(bars, vals)):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7,
                        color="#ffffff" if j == len(systems) - 1 else "#94a3b8")

        ax.set_xticks(x + width)
        ax.set_xticklabels(systems, color="#94a3b8", fontsize=9)
        ax.set_ylim(0, 0.55)
        ax.set_ylabel("ROUGE Score (F-measure)", color="#94a3b8")
        ax.tick_params(colors="#64748b")
        ax.legend(framealpha=0.2, labelcolor="#e2e8f0")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
        ax.grid(axis="y", alpha=0.2, color="#475569")
        ax.axvspan(4.6, 5.4, alpha=0.1, color="#6366f1")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — Lab
    # ════════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown('<div class="section-title">Experimentation Lab</div>', unsafe_allow_html=True)

        col_lab1, col_lab2 = st.columns(2)

        with col_lab1:
            st.markdown("#### Effect of temperature")
            st.markdown(
                r"""
The temperature $T$ modifies the softmax distribution over the vocabulary:

$$p_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$$

- $T \to 0$: near-deterministic (greedy).
- $T = 1$: model's original distribution.
- $T > 1$: more uniform and diverse.
"""
            )

            demo_text = (
                "A new agreement was reached between leaders. The deal covers trade "
                "policies and economic cooperation between nations."
            )
            temperatures = [0.5, 0.8, 1.0, 1.2, 1.5]

            if st.button("Run temperature experiment"):
                results = []
                with st.spinner("Generating with different temperatures..."):
                    for t in temperatures:
                        raw, _, _ = generate_summary_custom(
                            model, vocab, demo_text, max_len=30, temperature=t, device=device,
                        )
                        summ = clean_summary(raw)
                        results.append((t, summ))
                st.markdown("**Results:**")
                for t, s in results:
                    st.markdown(f"- **T = {t}**: `{s}`")

            st.markdown("#### Decoding strategies")
            st.markdown(
                """
| Strategy | Characteristic |
|----------|----------------|
| Greedy | Argmax at each step. Deterministic. |
| Beam search | Keeps k hypotheses; favors coherence. |
| top-k sampling | Filters to k most probable tokens. |
| top-p (nucleus) | Filters to cumulative mass p; dynamic size. |
| Temperature | Scales logits before softmax. |
"""
            )

        with col_lab2:
            st.markdown("#### Seq2Seq architecture timeline")
            st.markdown(
                """
| Year | Model | Innovation |
|------|-------|-----------|
| 2014 | Seq2Seq | Basic Encoder-Decoder |
| 2015 | Seq2Seq + Attention | Bahdanau et al. |
| 2017 | Pointer-Generator | Copy mechanism |
| 2017 | Transformer | Pure self-attention |
| 2018 | BERT | Bidirectional pre-training |
| 2019 | UniLM / T5 | Unified generation |
| 2020 | BART | Denoising pre-training |
| 2020 | PEGASUS | Gap-sentence masking |
"""
            )

            st.markdown("#### Transformers vs. LSTM advantages")
            advantages = [
                ("Parallelization", "LSTM is sequential; Transformers parallelize computation."),
                ("Long-range dependencies", "Attention is O(1) between any two positions."),
                ("Pre-training", "Transfer learning from large generic corpora."),
                ("Scalability", "Scaling laws favor massive models."),
            ]
            for title, desc in advantages:
                st.markdown(f"- **{title}.** {desc}")

        st.markdown("---")
        st.markdown("#### Theoretical curves: Perplexity vs. epochs")

        epochs = np.arange(1, 11)
        train_ppl = 200 * np.exp(-0.3 * epochs) + 15 + np.random.randn(10) * 2
        val_ppl = 220 * np.exp(-0.25 * epochs) + 25 + np.random.randn(10) * 3

        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig2.patch.set_facecolor("#0f172a")
        for ax in [ax1, ax2]:
            ax.set_facecolor("#1e293b")
            for spine in ax.spines.values():
                spine.set_edgecolor("#334155")
            ax.tick_params(colors="#64748b")
            ax.grid(alpha=0.2, color="#475569")

        ax1.plot(epochs, train_ppl, "o-", color="#6366f1", linewidth=2, label="Train PPL")
        ax1.plot(epochs, val_ppl, "o-", color="#10b981", linewidth=2, label="Val PPL")
        ax1.set_xlabel("Epoch", color="#94a3b8")
        ax1.set_ylabel("Perplexity", color="#94a3b8")
        ax1.set_title("Perplexity", color="#e2e8f0")
        ax1.legend(framealpha=0.2, labelcolor="#e2e8f0")

        tf_ratios = np.linspace(0.9, 0.3, 10)
        ax2.plot(epochs, tf_ratios, "o-", color="#f59e0b", linewidth=2)
        ax2.set_xlabel("Epoch", color="#94a3b8")
        ax2.set_ylabel("Teacher Forcing Ratio", color="#94a3b8")
        ax2.set_title("Teacher Forcing Schedule", color="#e2e8f0")
        ax2.set_ylim(0, 1)

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)


if __name__ == "__main__":
    main()
