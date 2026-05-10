# models.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer

# -------------------------
# Summarizer (facebook/bart-large-cnn)
# -------------------------
class Summarizer:
    _tokenizer = None
    _model = None

    @classmethod
    def get(cls):
        if cls._tokenizer is None or cls._model is None:
            cls._tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
            cls._model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
        return cls._tokenizer, cls._model

# -------------------------
# Paraphraser (t5)
# -------------------------
class Paraphraser:
    _tokenizer = None
    _model = None

    @classmethod
    def get(cls):
        if cls._tokenizer is None or cls._model is None:
            cls._tokenizer = AutoTokenizer.from_pretrained("ramsrigouthamg/t5_paraphraser")
            cls._model = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5_paraphraser")
        return cls._tokenizer, cls._model

# -------------------------
# Zero-shot classifier (BART MNLI)
# -------------------------
class ZeroShotClassifier:
    _pipeline = None

    @classmethod
    def get(cls):
        if cls._pipeline is None:
            cls._pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        return cls._pipeline

# -------------------------
# Sentence Transformer (MiniLM)
# -------------------------
class SentenceEmbedder:
    _model = None

    @classmethod
    def get(cls):
        if cls._model is None:
            cls._model = SentenceTransformer("all-MiniLM-L6-v2")
        return cls._model