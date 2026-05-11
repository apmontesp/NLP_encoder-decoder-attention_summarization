# Resumen Abstractivo de Noticias con Encoder-Decoder + Atención
## TensorFlow/Keras · SpaCy · CNN/DailyMail

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/apmontesp/NLP_encoder-decoder-attention_summarization/blob/main/CNN_DailyMail_EncoderDecoder_Attention.ipynb)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nlpencoder-decoder-attentionsummarization.streamlit.app/)

Taller de Procesamiento de Lenguaje Natural · Arquitectura Sequence-to-Sequence con mecanismo de atención sobre el dataset `cnn_dailymail` (Hugging Face Datasets, v3.0.0). El modelo opera **únicamente en inglés**.

---

## 1. Problema abordado

El dataset CNN/DailyMail es el benchmark de referencia para la tarea de **resumen abstractivo de texto**. Dado un artículo periodístico (≈ 780 palabras en promedio), el objetivo es generar un resumen abstractivo conciso (≈ 55 palabras) que sintetice la información saliente del documento.

A diferencia del resumen extractivo —que selecciona oraciones literales del texto fuente—, la formulación abstractiva exige al modelo: comprender el contenido global del artículo, identificar la información relevante, y generar texto nuevo, coherente y semánticamente fiel.

---

## 2. Arquitectura

El proyecto tiene dos implementaciones independientes que comparten la misma lógica conceptual pero difieren en framework y detalles técnicos.

### 2.1 Notebook (TensorFlow / Keras)

```
Artículo --> [Encoder LSTM + SpaCy 300d] --> {h_1, ..., h_T}
                                                    |
                                    [Atención de Luong (multiplicativa)] <-- s_t
                                                    |
                                         Vector de contexto c_t
                                                    |
                              [Concatenate(decoder_out, c_t)] --> [Dense softmax] --> token
```

| Componente | Detalle |
|-----------|---------|
| Encoder | LSTM unidireccional, dimensión oculta variable (`LATENT_DIM`) |
| Embeddings | `en_core_web_md` de SpaCy (300d preentrenados), `trainable=False` |
| Atención | Luong multiplicativa — capa `keras.layers.Attention` |
| Decoder | LSTM con teacher forcing; inicializado con estados del encoder |
| Vocabulario | 30 000 tokens en modo completo / 10 000 en FAST_MODE |
| Tokens especiales | `<start>` / `<end>` |
| Framework | TensorFlow / Keras |

### 2.2 Aplicación Streamlit (PyTorch)

```
Artículo --> [Encoder BiLSTM + GloVe 100d] --> {h_1, ..., h_T}
                                                      |
                                    [Atención aditiva de Bahdanau] <-- s_t
                                                      |
                                         Vector de contexto c_t
                                                      |
                                       [Decoder LSTM] --> [Linear] --> token
```

| Componente | Detalle |
|-----------|---------|
| Encoder | LSTM bidireccional, 2 capas, dimensión oculta 256 |
| Embeddings | GloVe 6B 100d, ajustables |
| Atención | Aditiva (Bahdanau) |
| Decoder | LSTM unidireccional, 2 capas |
| Vocabulario | 30 000 tokens (word-level) |
| Framework | PyTorch |

---

## 3. Estructura del repositorio

```
NLP_summarización_ML/
├── CNN_DailyMail_EncoderDecoder_Attention.ipynb   # Notebook principal (TF/Keras)
├── app.py                                         # Aplicación Streamlit (PyTorch)
├── requirements.txt
└── README.md
```

---

## 4. Modos de ejecución del notebook

El notebook soporta dos modos controlados por la variable `FAST_MODE` (celda 4):

| Parámetro | `FAST_MODE = True` | `FAST_MODE = False` |
|-----------|:-----------------:|:-------------------:|
| Muestras de entrenamiento | 2 000 | 50 000 |
| Épocas | 2 | 25 |
| `MAX_ARTICLE_LEN` | 100 tokens | 200 tokens |
| `MAX_SUMMARY_LEN` | 30 tokens | 60 tokens |
| `VOCAB_SIZE` | 10 000 | 30 000 |
| `LATENT_DIM` | 128 | 300 |
| Tiempo estimado (GPU Colab) | 5 – 10 min | 3 – 5 horas |
| Calidad de resúmenes | Baja (verificación técnica) | Aceptable académicamente |

> **Nota.** FAST_MODE sirve únicamente para verificar que todas las celdas corren sin error. Los resúmenes generados en este modo son incoherentes por insuficiencia de entrenamiento. Para resultados defendibles usar `FAST_MODE = False`.

---

## 5. Cache de artefactos en Google Drive

El notebook detecta automáticamente si existen artefactos previos en Drive y omite el entrenamiento en ese caso. El cache se invalida automáticamente si los hiperparámetros actuales (vocab size, longitudes máximas) difieren de los guardados, evitando errores de shape al cambiar entre FAST_MODE y modo completo.

**Artefactos generados:**

| Archivo | Contenido |
|---------|-----------|
| `nmt_model.h5` | Pesos del modelo Keras |
| `tokenizer.pkl` | Tokenizer con vocabulario |
| `embedding_matrix.npy` | Matriz de embeddings SpaCy |
| `model_config.pkl` | Hiperparámetros y scores ROUGE |

---

## 6. Ejecución

### 6.1 Notebook (Google Colab)

1. Abrir con el botón **Open In Colab**.
2. En la celda 4, configurar `FAST_MODE = False` para entrenamiento completo.
3. Ejecutar todas las celdas. Los artefactos se guardan automáticamente en Google Drive.
4. En sesiones posteriores el notebook detecta el cache y salta directamente a inferencia.

Para forzar reentrenamiento (por ejemplo, al cambiar `FAST_MODE`):
```python
FORCE_RETRAIN = True   # celda 2
```

### 6.2 Aplicación Streamlit (local)

```bash
pip install -r requirements.txt
# Opcional: coloque best_model.pt y vocab.pkl junto a app.py para usar el modelo entrenado.
streamlit run app.py
```

La app funciona en **modo demostración** si no se encuentran los pesos entrenados, usando un vocabulario reducido con pesos aleatorios. En ese caso se recomienda cambiar el backend a **BART preentrenado (Hugging Face)** desde la barra lateral.

---

## 7. Funcionalidades de la aplicación

| Pestaña | Funcionalidad |
|---------|--------------|
| **Summarizer** | Carga de archivos **PDF** o **TXT** en inglés, o entrada de texto pegado. Resumen abstractivo con el modelo entrenado o con DistilBART como referencia comparativa. |
| **Chat** | Conversación con historial persistente sobre el documento procesado. Preguntas frecuentes sobre contenido, entidades, atención y métricas ROUGE. |
| **Metrics** | Comparación cuantitativa con la literatura (Lead-3, Seq2Seq, Pointer-Generator, BART, PEGASUS) con visualización gráfica. |
| **Lab** | Sliders para ajustar temperatura, top-k, top-p, longitud máxima y número de beams. Experimento guiado sobre el efecto de la temperatura. |

> El modelo opera **únicamente en inglés**. No se realiza traducción de entrada ni de salida.

---

## 8. Resultados obtenidos

### 8.1 Cobertura léxica (OOV)

| Modo | Vocabulario | Tokens OOV (ejemplo CNN) |
|------|:-----------:|:------------------------:|
| FAST_MODE | 10 000 | 24.7 % |
| Completo | 30 000 | 12.3 % |

### 8.2 Calidad de resúmenes

Con **FAST_MODE** (2 épocas) el modelo no converge y genera tokens repetidos (`new . . . . .`). La app detecta este patrón y muestra un aviso explicativo.

Con el **modo completo** (25 épocas, 50 000 pares) el modelo genera oraciones con estructura gramatical válida. Se observa *hallucination* —el modelo produce texto plausible para el dominio de noticias pero no necesariamente fiel al artículo específico— comportamiento documentado y esperado en arquitecturas LSTM seq2seq sin mecanismo de cobertura.

### 8.3 ROUGE esperado (modo completo)

| Métrica | Modelo propuesto | Lead-3 (baseline) | BART (SOTA) |
|---------|:---------------:|:-----------------:|:-----------:|
| ROUGE-1 | 0.30 – 0.38 | 0.401 | 0.448 |
| ROUGE-2 | 0.12 – 0.16 | 0.175 | 0.214 |
| ROUGE-L | 0.28 – 0.35 | 0.365 | 0.412 |

---

## 9. Limitaciones conocidas

- Las arquitecturas LSTM presentan dificultades para modelar dependencias de largo alcance frente a Transformers.
- La tokenización word-level limita la cobertura léxica frente a sub-word tokenization (BPE / SentencePiece).
- El modelo no implementa mecanismo *pointer-generator*, por lo que tokens fuera de vocabulario se mapean a `<unk>`.
- Sin mecanismo de *coverage*, el decoder tiende a repetir frases al generar resúmenes largos.
- El vocabulario y los embeddings están optimizados para el dominio de noticias en inglés; textos de otros dominios producirán alta tasa OOV y resúmenes de menor calidad.

---

## 10. Referencias

1. *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR, 2015.
2. *Teaching Machines to Read and Comprehend.* NeurIPS, 2015.
3. *Get to the Point: Summarization with Pointer-Generator Networks.* ACL, 2017.
4. *Attention is All You Need.* NeurIPS, 2017.
5. *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation.* ACL, 2020.
6. *PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization.* ICML, 2020.
7. *GloVe: Global Vectors for Word Representation.* EMNLP, 2014.
8. *ROUGE: A Package for Automatic Evaluation of Summaries.* ACL Workshop, 2004.

---

## 11. Tecnologías

TensorFlow / Keras · PyTorch · SpaCy (`en_core_web_md`) · Hugging Face Datasets · Streamlit · Matplotlib · pypdf · rouge-score.
