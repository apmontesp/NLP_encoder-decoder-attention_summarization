# Resumen Automático de Noticias y Documentos
## Encoder-Decoder + Atención de Bahdanau · Embeddings GloVe

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/apmontesp/NLP_encoder-decoder-attention_summarization/blob/main/CNN_DailyMail_EncoderDecoder_Attention.ipynb)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nlpencoder-decoder-attentionsummarization.streamlit.app/)

Taller de Procesamiento de Lenguaje Natural · Arquitectura Sequence-to-Sequence con mecanismo de atención aditiva sobre el dataset `cnn_dailymail` (Hugging Face Datasets, v3.0.0).

---

## 1. Problema abordado

El dataset CNN/DailyMail (Hermann et al., 2015; See et al., 2017) es el benchmark de referencia para la tarea de **resumen abstractivo de texto**. Dado un artículo periodístico (≈ 780 palabras en promedio), el objetivo es generar un resumen abstractivo conciso (≈ 55 palabras) que sintetice la información saliente del documento.

A diferencia del resumen extractivo —que selecciona oraciones literales del texto fuente—, la formulación abstractiva exige al modelo: comprender el contenido global del artículo, identificar la información relevante, y generar texto nuevo, coherente y semánticamente fiel.

---

## 2. Arquitectura propuesta

```
Artículo --> [Encoder BiLSTM + GloVe] --> {h_1, ..., h_T}
                                                  |
                                  [Atención aditiva de Bahdanau] <-- s_t
                                                  |
                                       Vector de contexto c_t
                                                  |
                                     [Decoder LSTM] --> [Linear] --> token
```

| Componente | Descripción |
|-----------|-------------|
| Encoder | LSTM bidireccional, 2 capas, dimensión oculta 256 |
| Embeddings | GloVe 6B 100d (Pennington et al., 2014), ajustables |
| Atención | Aditiva (Bahdanau et al., 2015), proyección dim = 256 |
| Decoder | LSTM unidireccional, 2 capas, con `[embed; contexto]` como entrada |
| Vocabulario | 30 000 tokens (tokenización word-level) |

---

## 3. Estructura del repositorio

```
NLP_encoder-decoder-attention_summarization/
├── CNN_DailyMail_EncoderDecoder_Attention.ipynb   # Notebook principal
├── app.py                                         # Aplicación Streamlit
├── requirements.txt
└── README.md
```

---

## 4. Ejecución

### 4.1 Notebook (Google Colab)

Use el botón **Open In Colab** al inicio de este documento. El notebook ejecuta de extremo a extremo el flujo: carga del dataset, construcción del vocabulario, descarga de embeddings GloVe, entrenamiento, evaluación con ROUGE y persistencia de artefactos. Al finalizar se generan los archivos `best_model.pt`, `vocab.pkl` y `model_config.pkl`.

### 4.2 Aplicación Streamlit (local)

```bash
pip install -r requirements.txt
# Coloque best_model.pt y vocab.pkl en el directorio raíz si desea usar el modelo entrenado.
streamlit run app.py
```

Para despliegue en Streamlit Cloud, vincule el repositorio y especifique `app.py` como entrypoint.

---

## 5. Funcionalidades de la aplicación

| Pestaña | Funcionalidad |
|---------|--------------|
| Resumidor | Cargue de archivos **PDF**, **EPUB** o **TXT**, o entrada de texto pegado. Resumen abstractivo con el modelo entrenado o, opcionalmente, con BART preentrenado. |
| Chat interactivo | Conversación con historial persistente sobre el documento procesado. Incluye preguntas frecuentes y respuestas heurísticas sobre el modelo y la métrica ROUGE. |
| Métricas | Comparación cuantitativa con la literatura (Lead-3, Seq2Seq, Pointer-Generator, BART, PEGASUS) con visualización gráfica. |
| Laboratorio | Sliders para ajustar **temperatura**, **top-k**, **top-p**, **longitud máxima** y **número de beams**. Experimentos guiados sobre el efecto de la temperatura y notas sobre estrategias de decodificación. |

**Traducción.** El modelo se entrena sobre texto en inglés. La aplicación incorpora un módulo de traducción inglés → español basado en MarianMT (`Helsinki-NLP/opus-mt-en-es`) para entregar el resumen final en castellano cuando así se solicite.

---

## 6. Resultados esperados

| Métrica | Modelo propuesto | Lead-3 (baseline) | BART (SOTA) |
|---------|:----------------:|:-----------------:|:-----------:|
| ROUGE-1 | 0.30 – 0.38 | 0.401 | 0.448 |
| ROUGE-2 | 0.12 – 0.16 | 0.175 | 0.214 |
| ROUGE-L | 0.28 – 0.35 | 0.365 | 0.412 |

> Los rangos del modelo propuesto dependen del número de épocas de entrenamiento y del tamaño de la submuestra utilizada. La configuración por defecto del notebook (5 épocas, 50 000 ejemplos) sirve como punto de partida pedagógico, no como configuración óptima.

---

## 7. Limitaciones conocidas

- Las arquitecturas LSTM presentan dificultades para modelar dependencias de largo alcance frente a Transformers (Vaswani et al., 2017).
- La tokenización word-level limita la cobertura léxica frente a sub-word tokenization (BPE / SentencePiece).
- El modelo no implementa el mecanismo *pointer-generator* (See et al., 2017), por lo que las palabras fuera de vocabulario se rinden como `<unk>`.

---

## 8. Referencias

1. Bahdanau, D., Cho, K., & Bengio, Y. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR.
2. Hermann, K. M., Kočiský, T., Grefenstette, E., Espeholt, L., Kay, W., Suleyman, M., & Blunsom, P. (2015). *Teaching Machines to Read and Comprehend.* NeurIPS.
3. Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V., & Zettlemoyer, L. (2020). *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension.* ACL.
4. Lin, C.-Y. (2004). *ROUGE: A Package for Automatic Evaluation of Summaries.* ACL Workshop.
5. Pennington, J., Socher, R., & Manning, C. (2014). *GloVe: Global Vectors for Word Representation.* EMNLP.
6. See, A., Liu, P. J., & Manning, C. D. (2017). *Get to the Point: Summarization with Pointer-Generator Networks.* ACL.
7. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention is All You Need.* NeurIPS.
8. Zhang, J., Zhao, Y., Saleh, M., & Liu, P. J. (2020). *PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization.* ICML.

---

## 9. Tecnologías

PyTorch 2.x · Hugging Face Datasets · Hugging Face Transformers · MarianMT · Streamlit · Matplotlib · pypdf · EbookLib.
