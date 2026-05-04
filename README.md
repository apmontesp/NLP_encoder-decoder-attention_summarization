# CNN/DailyMail — Resumen Automático con Encoder-Decoder + Atención

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/apmontesp/NLP_encoder-decoder-attention_summarization/blob/main/CNN_DailyMail_EncoderDecoder_Attention.ipynb)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://apmontesp-nlp-encoder-decoder-attention-summarization.streamlit.app)

> **Taller de NLP** · Arquitectura Seq2Seq con Mecanismo de Atención de Bahdanau + Embeddings GloVe  
> Dataset: `cnn_dailymail` 3.0.0 — Hugging Face 

---

##  ¿Qué problema resuelve?

El dataset **CNN/DailyMail** es el benchmark estándar para **Resumen Abstractivo de Texto**:
dado un artículo periodístico (~800 palabras), generar un resumen abstractivo conciso (~55 palabras).

A diferencia del resumen extractivo (que copia frases del original), el resumen **abstractivo** requiere:
- Comprensión semántica profunda
- Generación de texto nuevo y coherente
- Captura de relaciones entre párrafos distantes

---

##  Arquitectura

```
Artículo → [Encoder BiLSTM + GloVe] → Estados ocultos
                                              ↓
                              [Atención Bahdanau] ← Hidden del Decoder
                                      ↓
                               Vector de Contexto
                                      ↓
                         [Decoder LSTM] → [Linear] → Token
```

| Componente | Detalle |
|-----------|---------|
| **Encoder** | LSTM Bidireccional (2 capas, hidden=256) |
| **Embeddings** | GloVe 6B 100d (preentrenados, fine-tunable) |
| **Atención** | Bahdanau Additive Attention |
| **Decoder** | LSTM Unidireccional (2 capas) |
| **Vocabulario** | 30,000 tokens |

---

##  Estructura del Repositorio

```
 NLP_encoder-decoder-attention_summarization/
├──  CNN_DailyMail_EncoderDecoder_Attention.ipynb  ← Notebook principal
├──  app.py                                         ← App Streamlit
├──  requirements.txt
└──  README.md
```

---

##  Cómo usar

### 1. Ejecutar el Notebook (Google Colab)
Haz click en el badge **Open in Colab** arriba. Ejecuta todas las celdas en orden.
Al finalizar, descarga `best_model.pt` y `vocab.pkl`.

### 2. Lanzar la App Streamlit

```bash
pip install -r requirements.txt
# Coloca best_model.pt y vocab.pkl en el directorio raíz
streamlit run app.py
```

O despliega en [Streamlit Cloud](https://streamlit.io/cloud) apuntando a `app.py`.

---

##  Resultados Esperados

| Métrica | Nuestro Modelo | Lead-3 (baseline) | BART (SOTA) |
|---------|:--------------:|:-----------------:|:-----------:|
| ROUGE-1 | ~0.30–0.38 | 0.401 | 0.448 |
| ROUGE-2 | ~0.12–0.16 | 0.175 | 0.214 |
| ROUGE-L | ~0.28–0.35 | 0.365 | 0.412 |

> Los resultados varían según el número de épocas y muestras de entrenamiento.

---

##  Referencias

1. **Bahdanau et al.** (2015) — *Neural Machine Translation by Jointly Learning to Align and Translate*
2. **See et al.** (2017) — *Get To The Point: Summarization with Pointer-Generator Networks*
3. **Hermann et al.** (2015) — *Teaching Machines to Read and Comprehend* (CNN/DailyMail original)
4. **Lewis et al.** (2020) — *BART: Denoising Sequence-to-Sequence Pre-training*
5. **Pennington et al.** (2014) — *GloVe: Global Vectors for Word Representation*
6. **Lin** (2004) — *ROUGE: A Package for Automatic Evaluation of Summaries*

---

## 🛠️ Tecnologías

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗_Hugging_Face-FFD21E?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python_3.10+-3776AB?style=flat&logo=python&logoColor=white)
