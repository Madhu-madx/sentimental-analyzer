# Sentiment Analysis with Hugging Face Transformers

This project implements a sentiment analysis pipeline using pretrained Transformer models. It is designed to handle different classification schemas (Binary and Ternary) and allows for flexible dataset sourcing from the Hugging Face Hub.

## 🎯 Project Objectives

* Load and utilize pretrained models/tokenizers for text classification.


* Perform inference to obtain predicted labels (POSITIVE, NEGATIVE, or NEUTRAL) and confidence scores.


* Support both default and manual Hugging Face dataset integration.


* Batch process multiple text inputs efficiently.



---

## 📊 Classification Capabilities

The notebook is configured to adapt to the specific model you choose from Hugging Face. Depending on the model's configuration, you can perform:

1. 
**Binary Classification:** Categorizing text as either **POSITIVE** or **NEGATIVE**.


2. 
**Ternary Classification:** Adding a **NEUTRAL** category for balanced or objective feedback.



---

## 📥 Data Sourcing from Hugging Face

You can load datasets using two primary methods:

### 1. Default Pipeline Loading

For quick testing, you can use the default Hugging Face pipeline, which automatically downloads a recommended model (typically `distilbert-base-uncased-finetuned-sst-2-english`).

```python
sentiment_analyzer = pipeline("sentiment-analysis")

```

### 2. Manual Link/Model Loading

If you require a specific model (e.g., a multilingual or ternary model), you can manually specify the repository link from Hugging Face.

**How to find a manual link:**

1. Visit [huggingface.co/models](https://huggingface.co/models).
2. Search for "sentiment-analysis".
3. Copy the model ID (e.g., `cardiffnlp/twitter-roberta-base-sentiment` for ternary).
4. Paste it into the `pipeline` or `AutoModel` initialization.



---

## 🛠️ Setup & Requirements

Before running the notebook, ensure the following dependencies are installed:

* 
`transformers`: For accessing Hugging Face APIs.


* 
`torch`: The PyTorch backend for model computation.


* 
`sentencepiece`: Required for specific multilingual models.



**Installation Command:**

```bash
pip install transformers torch sentencepiece

```

---

## 🚀 How to Use

1. 
**Initialize:** Run the import and pipeline setup cells.


2. 
**Define Inputs:** Provide a list of strings (e.g., product reviews or professor feedback).


3. 
**Run Inference:** The model will output the **label** and the **confidence score** (probability).


4. 
**Analyze:** Use the results to interpret the sentiment and score (e.g., a score of 0.98 indicates 98% confidence).



---

## 💡 Troubleshooting

* 
**Model Loading:** If the download fails, check your internet connection and verify the model name on Hugging Face.


* **Data Types:** Ensure all inputs in your list are strings; `None` or `NaN` values in a CSV will cause errors.

* 
**GPU Support:** For faster processing, verify your GPU status with `torch.cuda.is_available()`.
