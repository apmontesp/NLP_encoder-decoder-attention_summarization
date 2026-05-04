"""
NewsSum — Aplicación Streamlit
Resumen automático de noticias y documentos con Encoder-Decoder + Atención de Bahdanau.

Funcionalidades:
    1. Resumen abstractivo a partir de texto pegado, archivos PDF o archivos EPUB.
    2. Traducción opcional del resumen del inglés al español (MarianMT).
    3. Chat interactivo con historial persistente sobre el documento cargado.
    4. Panel de métricas con comparación frente a la literatura.
    5. Laboratorio para ajustar parámetros de inferencia y replicar experimentos.

Tono académico profesional. Solo se utiliza el carácter de validación U+2713.
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

CHECK = "✓"  # ✓


# ════════════════════════════════════════════════════════════════════════════
# Configuración general de la página
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NewsSum | Encoder-Decoder + Atención",
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
.main-title {
    font-size: 1.9rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0;
    letter-spacing: -0.4px;
}
.main-subtitle {
    color: #94a3b8;
    font-size: 0.92rem;
    margin-top: 0.4rem;
}
.tag {
    display: inline-block;
    background: rgba(99,102,241,0.18);
    color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.4);
    padding: 2px 10px;
    border-radius: 14px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 6px;
    margin-top: 8px;
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
    color: #818cf8;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 8px;
}

.section-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 0.9rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #334155;
}

.rouge-badge {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.3);
    padding: 4px 12px;
    border-radius: 8px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05rem;
}

.lit-table { width: 100%; border-collapse: collapse; }
.lit-table th {
    background: #1e293b;
    color: #94a3b8;
    font-size: 0.78rem;
    text-transform: uppercase;
    padding: 10px 14px;
    text-align: left;
}
.lit-table td {
    padding: 9px 14px;
    border-bottom: 1px solid #1e293b;
    color: #e2e8f0;
    font-size: 0.88rem;
}
.lit-table tr:hover td { background: #1e293b33; }
.lit-table .highlight td { color: #a5b4fc; font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)


# ════════════════════════════════════════════════════════════════════════════
# Utilidades de tokenización y vocabulario
# ════════════════════════════════════════════════════════════════════════════
def simple_tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return text.split()


class Vocabulary:
    """Vocabulario word-level con tokens especiales."""

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
# Arquitectura: Encoder + Atención + Decoder + Seq2Seq
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
        packed = pack_padded_sequence(
            embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False
        )
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
# Generación de resúmenes (modelo custom)
# ════════════════════════════════════════════════════════════════════════════
def generate_summary_custom(model, vocab, article_text, max_len=100,
                            temperature=1.0, top_k=0, top_p=0.0,
                            device=torch.device("cpu")):
    """Decodificación greedy / top-k / top-p sobre el modelo Encoder-Decoder."""
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

            # top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # top-p (nucleus) filtering
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
# Carga del modelo entrenado (con fallback de demostración)
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_custom_model():
    """Carga el modelo Encoder-Decoder entrenado o crea una versión de demostración."""
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


@st.cache_resource(show_spinner=False)
def load_hf_summarizer():
    """Pipeline preentrenado de Hugging Face (BART-large-cnn) — opcional."""
    try:
        from transformers import pipeline
        return pipeline("summarization", model="facebook/bart-large-cnn",
                        device=-1, framework="pt")
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_translator_en_es():
    """Traductor MarianMT EN → ES. Devuelve None si transformers no está disponible."""
    try:
        from transformers import pipeline
        return pipeline("translation_en_to_es",
                        model="Helsinki-NLP/opus-mt-en-es",
                        device=-1, framework="pt")
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_translator_es_en():
    """Traductor MarianMT ES → EN. Permite resumir documentos en español."""
    try:
        from transformers import pipeline
        return pipeline("translation",
                        model="Helsinki-NLP/opus-mt-es-en",
                        device=-1, framework="pt")
    except Exception:
        return None


# Heurística simple de detección de idioma (sin dependencias externas)
_SPANISH_HINTS = {
    "que", "para", "como", "pero", "más", "mas", "donde", "cuando",
    "porque", "esta", "está", "estaba", "fue", "fueron", "uno", "una",
    "del", "los", "las", "señor", "señora", "años", "día", "noche",
    "casa", "tiempo", "hombre", "mujer", "muy", "también", "sólo", "solo",
    "sino", "sin", "según", "según", "después", "antes", "entonces",
}
_ENGLISH_HINTS = {
    "the", "and", "of", "to", "in", "is", "was", "were", "are", "for",
    "with", "that", "this", "from", "have", "has", "had", "their", "they",
    "would", "could", "should", "where", "when", "while", "after", "before",
    "according", "between", "through", "however", "because",
}


def detect_language(text: str) -> str:
    """Devuelve 'es', 'en' o 'unknown' usando una heurística léxica.

    Cuenta palabras función características de cada idioma sobre la primera
    porción del documento. Suficiente para distinguir español de inglés;
    no pretende ser robusto frente a otros idiomas.
    """
    sample = text[:4000].lower()
    # Tokenización simple manteniendo letras Unicode
    import re as _re
    tokens = _re.findall(r"[a-záéíóúñü]+", sample, flags=_re.IGNORECASE)
    if len(tokens) < 5:
        return "unknown"
    es_hits = sum(1 for t in tokens if t in _SPANISH_HINTS)
    en_hits = sum(1 for t in tokens if t in _ENGLISH_HINTS)
    # Pista adicional: presencia de caracteres exclusivos del español
    has_spanish_chars = bool(_re.search(r"[áéíóúñ¡¿]", sample))
    if has_spanish_chars and es_hits >= en_hits:
        return "es"
    if es_hits > en_hits * 1.2:
        return "es"
    if en_hits > es_hits * 1.2:
        return "en"
    return "unknown"


# ════════════════════════════════════════════════════════════════════════════
# Extracción de texto desde archivos PDF y EPUB
# ════════════════════════════════════════════════════════════════════════════
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extrae texto plano de un PDF utilizando pypdf."""
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


def extract_text_from_epub(file_bytes: bytes) -> str:
    """Extrae texto plano de un EPUB usando ebooklib + BeautifulSoup."""
    try:
        from ebooklib import epub, ITEM_DOCUMENT
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError(
            "Para procesar EPUB se requieren los paquetes 'ebooklib' y 'beautifulsoup4'."
        ) from exc

    # ebooklib lee desde un path. Persistimos temporalmente.
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        book = epub.read_epub(tmp_path)
        chunks = []
        for item in book.get_items_of_type(ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), "html.parser")
            chunks.append(soup.get_text(separator=" ", strip=True))
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    return "\n".join(chunks).strip()


def extract_uploaded_text(uploaded_file) -> str:
    """Despacha la extracción según la extensión del archivo subido.

    Importante: se utiliza ``getvalue()`` (en lugar de ``read()``) para evitar
    que el cursor del UploadedFile quede al final tras la primera lectura y que
    los reruns posteriores de Streamlit obtengan bytes vacíos.
    """
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    if hasattr(uploaded_file, "getvalue"):
        data = uploaded_file.getvalue()
    else:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
        data = uploaded_file.read()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(data)
    if name.endswith(".epub"):
        return extract_text_from_epub(data)
    if name.endswith(".txt"):
        # Detección de codificación tolerante (UTF-8, latin-1)
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace")
    raise ValueError(f"Formato no soportado: {name}")


# ════════════════════════════════════════════════════════════════════════════
# Traducción inglés → español por trozos
# ════════════════════════════════════════════════════════════════════════════
def _translate_in_chunks(text: str, translator, max_chars: int = 350) -> str:
    """Traduce texto largo dividiéndolo por oraciones para no exceder el
    límite de 512 tokens de los modelos MarianMT."""
    if translator is None or not text.strip():
        return ""
    sentences = re.split(r"(?<=[\.\!\?])\s+", text.strip())
    out = []
    buffer = ""
    for s in sentences:
        if len(buffer) + len(s) < max_chars:
            buffer = (buffer + " " + s).strip()
        else:
            if buffer:
                out.append(translator(buffer)[0]["translation_text"])
            buffer = s
    if buffer:
        out.append(translator(buffer)[0]["translation_text"])
    return " ".join(out)


def translate_en_to_es(text: str, translator) -> str:
    """Traduce un texto del inglés al español por trozos."""
    return _translate_in_chunks(text, translator)


def translate_es_to_en(text: str, translator) -> str:
    """Traduce un texto del español al inglés por trozos.

    Para entradas muy largas (libros enteros, p. ej. *Cien años de soledad*)
    el modelo MarianMT puede tardar minutos. Se acota la entrada a los
    primeros 12 000 caracteres para mantener la app responsiva.
    """
    return _translate_in_chunks(text[:12000], translator)


# ════════════════════════════════════════════════════════════════════════════
# Generación con BART (HF) — opcional
# ════════════════════════════════════════════════════════════════════════════
def generate_summary_bart(text: str, summarizer, max_length=130, min_length=30,
                          num_beams=4) -> str:
    """Resumen con BART-large-cnn. Trunca a 1024 tokens (límite del modelo)."""
    if summarizer is None:
        return ""
    text = text[:6000]  # truncado por seguridad
    out = summarizer(text, max_length=max_length, min_length=min_length,
                     num_beams=num_beams, truncation=True)
    return out[0]["summary_text"]


# ════════════════════════════════════════════════════════════════════════════
# Respuestas del chat
# ════════════════════════════════════════════════════════════════════════════
def generate_chat_response(question: str, article: str, summary: str) -> str:
    """Respuesta basada en reglas y extracción heurística sobre el artículo."""
    q = question.lower().strip()

    if any(w in q for w in ["de qué", "de que", "trata", "tema", "about"]):
        return (f"El documento se sintetiza así: <i>{summary}</i>"
                if summary else "Aún no hay un resumen disponible. Genera uno primero.")

    if any(w in q for w in ["quién", "quien", "personas", "entidad", "mencion"]):
        if not article:
            return "No hay un artículo cargado."
        caps = [w for w in article.split() if w[:1].isupper() and len(w) > 2 and w.isalpha()]
        unique = list(dict.fromkeys(caps))[:10]
        return ("Entidades / nombres detectados: <b>" + ", ".join(unique) + "</b>"
                if unique else "No se identificaron entidades claras.")

    if any(w in q for w in ["atencion", "atención", "attention", "como funciona", "cómo funciona"]):
        return (
            "<b>Mecanismo de Atención de Bahdanau:</b><br>"
            "En cada paso del decoder se computa un score de alineación "
            "<code>e(t,i) = v · tanh(W_enc h_i + W_dec s_t)</code>, "
            "se aplica softmax para obtener pesos <code>α</code> y se calcula el "
            "vector de contexto como suma ponderada de los estados del encoder. "
            "Esto permite enfocar regiones relevantes del documento al generar "
            "cada token del resumen."
        )

    if any(w in q for w in ["rouge", "métrica", "metrica", "evaluacion", "evaluación"]):
        return (
            "<b>ROUGE</b> mide solapamiento de n-gramas y subsecuencias entre el "
            "resumen generado y el de referencia. Se reportan ROUGE-1 (unigramas), "
            "ROUGE-2 (bigramas) y ROUGE-L (subsecuencia común más larga). "
            "En CNN/DailyMail, valores superiores a 0.35 en ROUGE-1 son aceptables "
            "para arquitecturas Encoder-Decoder clásicas."
        )

    if any(w in q for w in ["temperatura", "temperature"]):
        return (
            "La <b>temperatura</b> escala los logits antes del softmax: "
            "T &lt; 1 produce salidas más determinísticas, T &gt; 1 introduce más "
            "diversidad. Para resúmenes informativos se recomienda T ∈ [0.7, 1.0]."
        )

    if any(w in q for w in ["hola", "buenas", "buenos", "hello", "hi"]):
        return (
            "Hola. Soy NewsSum. Carga un artículo o documento, genera el resumen "
            "y luego puedes preguntarme por su contenido, las métricas o la "
            "arquitectura del modelo."
        )

    # Búsqueda heurística en el artículo
    if article:
        sentences = re.split(r"(?<=[\.\!\?])\s+", article)
        keywords = [w for w in q.split() if len(w) > 3]
        relevant = [s for s in sentences
                    if any(k in s.lower() for k in keywords)]
        if relevant:
            return ("Pasaje relevante en el documento: "
                    f"<i>«{relevant[0][:280].strip()}…»</i>")
    return (f"Pregunta registrada: <i>{question}</i>. Puedo ayudarte con resúmenes, "
            "entidades del documento, métricas ROUGE o la arquitectura del modelo.")


# ════════════════════════════════════════════════════════════════════════════
# Render del mapa de atención
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
    ax.set_xticklabels(src_tokens[:n_src], rotation=45, ha="right",
                       fontsize=8, color="#94a3b8")
    ax.set_yticklabels(trg_tokens[:n_trg], fontsize=9, color="#e2e8f0")
    ax.set_xlabel("Documento (tokens fuente)", color="#94a3b8")
    ax.set_ylabel("Resumen (tokens objetivo)", color="#94a3b8")
    ax.tick_params(colors="#64748b")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(colors="#94a3b8")
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════════
# Aplicación principal
# ════════════════════════════════════════════════════════════════════════════
def main():
    model, vocab, device, model_loaded = load_custom_model()

    # ── Encabezado ──────────────────────────────────────────────────────────
    status_tag = (
        f'<span class="tag green">{CHECK} Modelo entrenado cargado</span>'
        if model_loaded
        else '<span class="tag amber">Modo demostración — pesos aleatorios</span>'
    )
    st.markdown(
        f"""
    <div class="main-header">
      <p class="main-title">NewsSum</p>
      <p class="main-subtitle">
        Resumen automático de noticias y documentos · Encoder-Decoder + Atención de Bahdanau · GloVe 6B
      </p>
      <span class="tag">CNN/DailyMail v3.0.0</span>
      <span class="tag">PyTorch</span>
      <span class="tag green">Seq2Seq</span>
      <span class="tag green">Atención aditiva</span>
      {status_tag}
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Aviso prominente cuando no hay checkpoint entrenado
    if not model_loaded:
        st.warning(
            "**Modo demostración activo.** El archivo `best_model.pt` no se "
            "encontró en el directorio, así que el backend *Modelo entrenado* "
            "está operando con pesos aleatorios y un vocabulario reducido — los "
            "resúmenes serán secuencias de palabras sin sentido. Para obtener "
            "resultados reales: (1) ejecute el notebook hasta el final y copie "
            "`best_model.pt` y `vocab.pkl` junto a `app.py`, o (2) cambie el "
            "backend en la barra lateral a **BART preentrenado (Hugging Face)**."
        )

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## Laboratorio de parámetros")

        st.markdown("### Modelo de inferencia")
        backend = st.radio(
            "Backend",
            options=["Modelo entrenado (Encoder-Decoder + Atención)",
                     "BART preentrenado (Hugging Face)"],
            help=("El modelo entrenado replica la arquitectura del taller. "
                  "BART preentrenado se incluye como referencia comparativa."),
        )

        st.markdown("### Decodificación")
        temperature = st.slider("Temperatura", 0.05, 2.0, 1.0, 0.05,
                                help="Escala los logits antes del softmax.")
        top_k = st.slider("top-k", 0, 200, 0, 5,
                          help="Filtra a los k tokens más probables. 0 = desactivado.")
        top_p = st.slider("top-p (nucleus)", 0.0, 1.0, 0.0, 0.05,
                          help="Filtra a la masa de probabilidad acumulada p. 0 = desactivado.")
        max_len = st.slider("Longitud máxima del resumen (tokens)", 30, 200, 80, 5)
        num_beams = st.slider("Beams (sólo BART)", 1, 8, 4, 1)

        st.markdown("### Idioma")
        output_language = st.radio(
            "Idioma del resumen",
            options=["Español", "Inglés"],
            index=0,
            horizontal=True,
            help=("Idioma en el que se mostrará el resumen final. El modelo "
                  "opera internamente en inglés; si elige español el resumen "
                  "se traduce con Helsinki-NLP/opus-mt-en-es."),
        )
        auto_translate_input = st.checkbox(
            "Traducir entrada automáticamente al inglés si está en español",
            value=True,
            help=("El modelo se entrena con CNN/DailyMail en inglés. Esta opción "
                  "detecta el idioma del documento y, si es español, lo traduce "
                  "al inglés antes de resumir. Utiliza Helsinki-NLP/opus-mt-es-en."),
        )

        st.markdown("### Visualización")
        show_attention = st.checkbox("Mostrar mapa de atención (modelo entrenado)", True)
        show_rouge = st.checkbox("Calcular ROUGE contra el documento original", True)

        st.markdown("---")
        st.markdown("### Configuración del modelo")
        st.code(
            f"""Encoder   : BiLSTM (2 capas)
Decoder   : LSTM   (2 capas)
Atención  : Bahdanau aditiva
Embedding : GloVe 6B 100d
Hidden    : 256
Vocab     : {len(vocab):,}""",
            language="text",
        )

        st.markdown("### Literatura recomendada")
        st.markdown(
            """
- Bahdanau et al. (2015) — Atención aditiva.
- See et al. (2017) — Pointer-Generator + Coverage.
- Lewis et al. (2020) — BART.
- Zhang et al. (2020) — PEGASUS.
- Lin (2004) — ROUGE.
"""
        )

    # ── Pestañas ────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Resumidor", "Chat interactivo", "Métricas", "Laboratorio"]
    )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — Resumidor
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown('<div class="section-title">Generador de resúmenes</div>',
                    unsafe_allow_html=True)

        col1, col2 = st.columns([3, 2])

        with col1:
            # ── Ejemplos predefinidos (deben evaluarse antes del text_area
            # para poder escribir en st.session_state["article_input"]) ──
            example_articles = {
                "Política": (
                    "The president announced a sweeping new climate policy that would "
                    "require all federal buildings to be powered by renewable energy by "
                    "2030. The executive order signed on Thursday also establishes a new "
                    "task force to oversee the transition and allocate funding from the "
                    "infrastructure bill. Critics from the energy sector warned the plan "
                    "could increase utility costs, while environmental groups praised it "
                    "as an important step toward reducing carbon emissions."
                ),
                "Economía": (
                    "Global stock markets tumbled on Monday as investors reacted to new "
                    "inflation data showing prices rose faster than expected last month. "
                    "The consumer price index climbed 8.5 percent year over year, "
                    "surpassing economist forecasts and renewing fears that the Federal "
                    "Reserve may need to raise interest rates more aggressively."
                ),
                "Ciencia": (
                    "Scientists announced a breakthrough in fusion energy research after "
                    "achieving a net energy gain for the second time at the National "
                    "Ignition Facility in California. The experiment produced 3.15 "
                    "megajoules of energy from a 2.05 megajoule laser input, marking a "
                    "significant milestone in the decades-long quest for clean limitless "
                    "power."
                ),
            }

            # ── Cargue de archivos (PDF / EPUB / TXT) ──
            uploaded_file = st.file_uploader(
                "Cargar documento (PDF, EPUB o TXT en inglés)",
                type=["pdf", "epub", "txt"],
                help=("El texto se procesará en inglés. Si activa la traducción, "
                      "el resumen se entregará también en español."),
                key="uploader",
            )

            # Detectar nuevos cargues por (nombre, tamaño) para no reextraer
            # en cada rerun y para escribir el contenido en session_state
            # ANTES de instanciar el text_area.
            if uploaded_file is not None:
                file_id = (uploaded_file.name, getattr(uploaded_file, "size", None))
                if st.session_state.get("_last_uploaded_id") != file_id:
                    try:
                        with st.spinner(
                            f"Extrayendo texto de {uploaded_file.name} ..."
                        ):
                            extracted_text = extract_uploaded_text(uploaded_file)
                        if extracted_text and extracted_text.strip():
                            st.session_state["article_input"] = extracted_text
                            st.session_state["_last_uploaded_id"] = file_id
                            st.success(
                                f"{CHECK} {len(extracted_text.split()):,} palabras "
                                f"extraídas de {uploaded_file.name}."
                            )
                        else:
                            st.warning(
                                "No se pudo extraer texto del archivo "
                                "(documento vacío o ilegible)."
                            )
                    except Exception as exc:
                        st.error(f"Error al procesar el archivo: {exc}")

            # Botones de ejemplo (escriben directamente en session_state
            # antes de que el text_area se renderice).
            st.markdown("**Ejemplos de referencia:**")
            ex1, ex2, ex3 = st.columns(3)
            if ex1.button("Política"):
                st.session_state["article_input"] = example_articles["Política"]
                st.session_state["_last_uploaded_id"] = None
                st.rerun()
            if ex2.button("Economía"):
                st.session_state["article_input"] = example_articles["Economía"]
                st.session_state["_last_uploaded_id"] = None
                st.rerun()
            if ex3.button("Ciencia"):
                st.session_state["article_input"] = example_articles["Ciencia"]
                st.session_state["_last_uploaded_id"] = None
                st.rerun()

            # ── text_area gobernado por session_state ──
            # No se pasa `value=`: cuando hay `key=`, Streamlit toma el
            # contenido de st.session_state[key]. Así, el texto extraído del
            # archivo y los ejemplos se reflejan correctamente en el widget.
            article_input = st.text_area(
                "O pegue el texto a resumir aquí (en inglés):",
                height=260,
                placeholder=(
                    "Ingrese el artículo o pegue el contenido extraído del archivo. "
                    "Ejemplo: The president announced a new climate policy ..."
                ),
                key="article_input",
            )

            generate_btn = st.button(
                "Generar resumen", type="primary", use_container_width=True
            )

        with col2:
            if st.session_state.get("last_summary"):
                summary, attn_weights, src_ids = st.session_state["last_summary"]
                summary_es = st.session_state.get("last_summary_es", "")
                lang_choice = st.session_state.get(
                    "last_summary_lang", output_language
                )

                # Resumen principal en el idioma elegido por el usuario
                if lang_choice == "Español" and summary_es:
                    primary_label = "Resumen (español)"
                    primary_text = summary_es
                    primary_border = "#10b981"
                    primary_label_color = "#6ee7b7"
                    secondary_label = "Versión original generada por el modelo (inglés)"
                    secondary_text = summary
                else:
                    primary_label = "Resumen (inglés)"
                    primary_text = summary
                    primary_border = "#6366f1"
                    primary_label_color = "#818cf8"
                    secondary_label = ""
                    secondary_text = ""

                st.markdown(
                    f"""
                <div class="summary-box" style="border-left-color:{primary_border};">
                  <div class="summary-label" style="color:{primary_label_color};">{primary_label}</div>
                  <div class="summary-text">{primary_text}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # Versión secundaria (solo cuando el principal está en español):
                # mostramos también el inglés en un expander para fines académicos.
                if secondary_text:
                    with st.expander(secondary_label, expanded=False):
                        st.markdown(
                            f'<div class="summary-text">{secondary_text}</div>',
                            unsafe_allow_html=True,
                        )

                src_words = len(article_input.split()) if article_input else 0
                trg_words = len(summary.split())
                ratio = src_words / max(trg_words, 1)
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Palabras (origen)", f"{src_words:,}")
                mc2.metric("Palabras (resumen)", f"{trg_words:,}")
                mc3.metric("Compresión", f"{ratio:.1f}x")

                if show_rouge and article_input and summary:
                    try:
                        scorer = rouge_scorer.RougeScorer(
                            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
                        )
                        sc = scorer.score(article_input[:1500], summary)
                        st.markdown("**ROUGE (resumen vs. documento):**")
                        rc1, rc2, rc3 = st.columns(3)
                        rc1.markdown(
                            f'<div class="rouge-badge">R-1: {sc["rouge1"].fmeasure:.3f}</div>',
                            unsafe_allow_html=True,
                        )
                        rc2.markdown(
                            f'<div class="rouge-badge">R-2: {sc["rouge2"].fmeasure:.3f}</div>',
                            unsafe_allow_html=True,
                        )
                        rc3.markdown(
                            f'<div class="rouge-badge">R-L: {sc["rougeL"].fmeasure:.3f}</div>',
                            unsafe_allow_html=True,
                        )
                    except Exception:
                        pass

                # Persistir el contexto del documento para el chat
                ctx = st.session_state.setdefault("chat_context", {})
                ctx["last_article"] = article_input
                ctx["last_summary"] = summary
                ctx["last_summary_es"] = st.session_state.get("last_summary_es", "")

        # Acción del botón
        if generate_btn and article_input and len(article_input.strip()) > 20:
            try:
                # ── (1) Detección de idioma y traducción ES → EN si aplica ──
                lang = detect_language(article_input)
                input_for_model = article_input
                if lang == "es" and auto_translate_input:
                    es_en = load_translator_es_en()
                    if es_en is None:
                        st.warning(
                            "Se detectó español pero no fue posible cargar el "
                            "traductor ES → EN. Instale 'transformers' y "
                            "'sentencepiece'. Continuando con el texto original."
                        )
                    else:
                        with st.spinner(
                            f"Texto detectado en español. Traduciendo a inglés "
                            f"antes de resumir ..."
                        ):
                            input_for_model = translate_es_to_en(
                                article_input, es_en
                            )
                        st.info(
                            f"{CHECK} Entrada traducida al inglés "
                            f"({len(input_for_model.split()):,} palabras)."
                        )
                elif lang == "es" and not auto_translate_input:
                    st.warning(
                        "El texto parece estar en español, pero la traducción "
                        "automática ES → EN está desactivada. El modelo está "
                        "entrenado en inglés; los resultados pueden ser pobres."
                    )

                # ── (2) Resumen ──
                if backend.startswith("BART"):
                    summarizer = load_hf_summarizer()
                    if summarizer is None:
                        st.error(
                            "No fue posible cargar BART. Verifique la instalación "
                            "de 'transformers'."
                        )
                        st.stop()
                    with st.spinner("Generando resumen con BART preentrenado ..."):
                        summary_en = generate_summary_bart(
                            input_for_model, summarizer,
                            max_length=max_len,
                            min_length=max(20, max_len // 4),
                            num_beams=num_beams,
                        )
                        st.session_state["last_summary"] = (summary_en, [], [])
                else:
                    if not model_loaded:
                        st.warning(
                            "Está usando el backend *Modelo entrenado* en modo "
                            "demostración (pesos aleatorios). El resumen carecerá "
                            "de sentido. Cambie a BART en la barra lateral."
                        )
                    with st.spinner(
                        "Generando resumen con el modelo Encoder-Decoder ..."
                    ):
                        summary_en, attn_weights, src_ids = generate_summary_custom(
                            model, vocab, input_for_model,
                            max_len=max_len, temperature=temperature,
                            top_k=top_k, top_p=top_p, device=device,
                        )
                        st.session_state["last_summary"] = (
                            summary_en, attn_weights, src_ids
                        )

                # Persistir el documento original (no el traducido) para el chat
                st.session_state["last_article_text"] = article_input

                # ── (3) Traducción del resumen al idioma seleccionado ──
                st.session_state["last_summary_lang"] = output_language
                if output_language == "Español":
                    en_es = load_translator_en_es()
                    if en_es is None:
                        st.warning(
                            "No fue posible cargar el traductor MarianMT EN → ES. "
                            "Instale 'transformers' y 'sentencepiece' para "
                            "habilitarlo. Mostrando el resumen en inglés."
                        )
                        st.session_state["last_summary_es"] = ""
                    else:
                        with st.spinner("Traduciendo resumen al español ..."):
                            st.session_state["last_summary_es"] = translate_en_to_es(
                                summary_en, en_es
                            )
                else:
                    # Idioma de salida = Inglés: no se requiere traducción
                    st.session_state["last_summary_es"] = ""

                st.rerun()
            except Exception as exc:
                st.error(f"Error durante la generación: {exc}")

        # Mapa de atención
        if (show_attention
                and backend.startswith("Modelo entrenado")
                and st.session_state.get("last_summary")):
            summary, attn_weights, _ = st.session_state["last_summary"]
            article_text = st.session_state.get("last_article_text", "")
            if attn_weights and article_text:
                st.markdown("---")
                st.markdown('<div class="section-title">Mapa de atención</div>',
                            unsafe_allow_html=True)
                fig = render_attention_map(article_text, summary, attn_weights)
                if fig is not None:
                    st.pyplot(fig)
                    plt.close(fig)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — Chat interactivo (con historial persistente)
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown('<div class="section-title">Chat interactivo</div>',
                    unsafe_allow_html=True)

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
        if "chat_context" not in st.session_state:
            st.session_state["chat_context"] = {}

        ctx = st.session_state["chat_context"]
        if not ctx.get("last_article"):
            st.info(
                "Genere primero un resumen en la pestaña **Resumidor** "
                "para habilitar el chat contextual."
            )
        else:
            st.success(
                f"{CHECK} Documento activo: {len(ctx['last_article'].split()):,} palabras."
            )

        # Render histórico con la API moderna st.chat_message
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"], unsafe_allow_html=True)
                if msg.get("time"):
                    st.caption(msg["time"])

        # Atajos de preguntas frecuentes
        st.markdown("**Preguntas frecuentes:**")
        q1, q2, q3, q4 = st.columns(4)
        quick = {
            "tema": "¿De qué trata el documento?",
            "entidades": "¿Qué personas o entidades se mencionan?",
            "atencion": "¿Cómo funciona el mecanismo de atención?",
            "rouge": "¿Cómo se interpreta la métrica ROUGE?",
        }
        triggered = None
        if q1.button(quick["tema"]):     triggered = quick["tema"]
        if q2.button(quick["entidades"]): triggered = quick["entidades"]
        if q3.button(quick["atencion"]):  triggered = quick["atencion"]
        if q4.button(quick["rouge"]):     triggered = quick["rouge"]

        # Entrada de chat
        user_input = st.chat_input("Escriba su pregunta sobre el documento ...")
        new_msg = triggered or user_input

        if new_msg:
            ts = time.strftime("%H:%M")
            st.session_state["chat_history"].append(
                {"role": "user", "content": new_msg, "time": ts}
            )
            response = generate_chat_response(
                new_msg,
                ctx.get("last_article", ""),
                ctx.get("last_summary", ""),
            )
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": response, "time": ts}
            )
            st.rerun()

        if st.session_state["chat_history"]:
            if st.button("Limpiar historial"):
                st.session_state["chat_history"] = []
                st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — Métricas
    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-title">Métricas de evaluación</div>',
                    unsafe_allow_html=True)

        col_a, col_b = st.columns([2, 1])

        with col_a:
            st.markdown("#### Comparación con la literatura")
            st.markdown(
                f"""
                <table class="lit-table">
                  <thead>
                    <tr><th>Modelo</th><th>Tipo</th><th>ROUGE-1</th><th>ROUGE-2</th><th>ROUGE-L</th><th>Año</th></tr>
                  </thead>
                  <tbody>
                    <tr><td>Lead-3</td><td>Extractivo (baseline)</td><td>0.401</td><td>0.175</td><td>0.365</td><td>—</td></tr>
                    <tr><td>Seq2Seq</td><td>Abstractivo básico</td><td>0.358</td><td>0.144</td><td>0.330</td><td>2015</td></tr>
                    <tr><td>Seq2Seq + Atención</td><td>Abstractivo</td><td>0.374</td><td>0.158</td><td>0.346</td><td>2015</td></tr>
                    <tr><td>Pointer-Generator</td><td>Abstractivo + Coverage</td><td>0.398</td><td>0.173</td><td>0.367</td><td>2017</td></tr>
                    <tr><td>UniLM</td><td>Transformer fine-tuned</td><td>0.435</td><td>0.203</td><td>0.402</td><td>2019</td></tr>
                    <tr><td>PEGASUS</td><td>Pre-entrenado abstractivo</td><td>0.447</td><td>0.214</td><td>0.417</td><td>2020</td></tr>
                    <tr><td>BART</td><td>Pre-entrenado abstractivo</td><td>0.448</td><td>0.214</td><td>0.412</td><td>2020</td></tr>
                    <tr class="highlight"><td>Modelo propuesto {CHECK}</td><td>Encoder-Decoder + GloVe</td><td>~0.30–0.38</td><td>~0.12–0.16</td><td>~0.28–0.35</td><td>2024</td></tr>
                  </tbody>
                </table>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
> **Nota.** Los modelos *state-of-the-art* (BART, PEGASUS) emplean
> arquitecturas Transformer con cientos de millones de parámetros y
> pre-entrenamiento masivo. El modelo propuesto en este taller cumple un rol
> pedagógico: ilustra los fundamentos del paradigma Encoder-Decoder con
> mecanismo de atención.
"""
            )

        with col_b:
            st.markdown("#### Interpretación de ROUGE")
            st.markdown(
                """
**ROUGE-N** mide solapamiento de n-gramas entre el resumen generado
y el de referencia:

```
ROUGE-1 : unigramas
ROUGE-2 : bigramas
ROUGE-L : subsecuencia común más larga
```

| Score | Calidad |
|-------|---------|
| > 0.40 | Excelente |
| 0.30 – 0.40 | Bueno |
| 0.20 – 0.30 | Aceptable |
| < 0.20 | Bajo |

**Métricas complementarias:** BERTScore (semántica), METEOR (morfología),
BLEU (n-gramas), Perplexity (entrenamiento).
"""
            )

        st.markdown("---")
        st.markdown("#### Comparación gráfica de ROUGE")

        systems = ["Lead-3", "Seq2Seq\nbásico", "Seq2Seq\n+ Atención",
                   "Pointer-Gen", "BART", "Propuesto"]
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
            bars = ax.bar(
                x + i * width, vals, width, label=label, color=color,
                alpha=0.85, edgecolor="#0f172a",
            )
            for j, (bar, v) in enumerate(zip(bars, vals)):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#ffffff" if j == len(systems) - 1 else "#94a3b8",
                )

        ax.set_xticks(x + width)
        ax.set_xticklabels(systems, color="#94a3b8", fontsize=9)
        ax.set_ylim(0, 0.55)
        ax.set_ylabel("Score ROUGE (F-measure)", color="#94a3b8")
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
    # TAB 4 — Laboratorio
    # ════════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown('<div class="section-title">Laboratorio de experimentación</div>',
                    unsafe_allow_html=True)

        col_lab1, col_lab2 = st.columns(2)

        with col_lab1:
            st.markdown("#### Efecto de la temperatura")
            st.markdown(
                """
La temperatura $T$ modifica la distribución softmax sobre el vocabulario:

$$p_i = \\frac{\\exp(z_i/T)}{\\sum_j \\exp(z_j/T)}$$

- $T \\to 0$: salida casi determinística (greedy).
- $T = 1$: distribución original del modelo.
- $T > 1$: distribución más uniforme y diversa.
"""
            )

            demo_text = (
                "A new agreement was reached between leaders. The deal covers trade "
                "policies and economic cooperation between nations."
            )
            temperatures = [0.5, 0.8, 1.0, 1.2, 1.5]

            if st.button("Ejecutar experimento de temperatura"):
                results = []
                with st.spinner("Generando con distintas temperaturas ..."):
                    for t in temperatures:
                        summ, _, _ = generate_summary_custom(
                            model, vocab, demo_text, max_len=30, temperature=t,
                            device=device,
                        )
                        results.append((t, summ))
                st.markdown("**Resultados del experimento:**")
                for t, s in results:
                    st.markdown(f"- **T = {t}**: `{s}`")

            st.markdown("#### Estrategias de decodificación")
            st.markdown(
                """
| Estrategia | Característica |
|-----------|----------------|
| Greedy | Toma el argmax en cada paso. Determinística. |
| Beam search | Mantiene k hipótesis simultáneas; favorece coherencia. |
| top-k sampling | Filtra a los k tokens más probables. |
| top-p (nucleus) | Filtra a la masa acumulada p; tamaño dinámico. |
| Temperatura | Escala los logits antes del softmax. |
"""
            )

        with col_lab2:
            st.markdown("#### Línea de tiempo de arquitecturas Seq2Seq")
            st.markdown(
                """
| Año | Modelo | Innovación |
|-----|--------|-----------|
| 2014 | Seq2Seq | Encoder-Decoder básico |
| 2015 | Seq2Seq + Atención | Bahdanau et al. |
| 2017 | Pointer-Generator | Mecanismo de copia |
| 2017 | Transformer | Self-attention puro |
| 2018 | BERT | Pre-entrenamiento bidireccional |
| 2019 | UniLM / T5 | Generación unificada |
| 2020 | BART | Denoising pre-training |
| 2020 | PEGASUS | Gap-sentence masking |
"""
            )

            st.markdown("#### Ventajas de los Transformers sobre LSTM")
            advantages = [
                ("Paralelización", "El LSTM es secuencial; el Transformer paraleliza el cálculo."),
                ("Dependencias largas", "Atención O(1) entre cualesquiera dos posiciones."),
                ("Pre-entrenamiento", "Transferencia desde grandes corpus genéricos."),
                ("Escalabilidad", "Las leyes de escala favorecen modelos masivos."),
            ]
            for title, desc in advantages:
                st.markdown(f"- **{title}.** {desc}")

        st.markdown("---")
        st.markdown("#### Curvas teóricas: Perplexity vs. épocas")

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
        ax1.set_xlabel("Época", color="#94a3b8")
        ax1.set_ylabel("Perplexity", color="#94a3b8")
        ax1.set_title("Perplexity", color="#e2e8f0")
        ax1.legend(framealpha=0.2, labelcolor="#e2e8f0")

        tf_ratios = np.linspace(0.9, 0.3, 10)
        ax2.plot(epochs, tf_ratios, "o-", color="#f59e0b", linewidth=2)
        ax2.set_xlabel("Época", color="#94a3b8")
        ax2.set_ylabel("Teacher Forcing Ratio", color="#94a3b8")
        ax2.set_title("Programa de Teacher Forcing", color="#e2e8f0")
        ax2.set_ylim(0, 1)

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)


if __name__ == "__main__":
    main()
