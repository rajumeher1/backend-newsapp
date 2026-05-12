# models.py

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)

from sentence_transformers import SentenceTransformer


# =========================================================
# Summarizer (BART CNN)
# =========================================================
class Summarizer:

    _tokenizer = None
    _model = None
    _device = None

    @classmethod
    def get(cls):

        if cls._tokenizer is None or cls._model is None:

            cls._device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

            cls._tokenizer = AutoTokenizer.from_pretrained(
                "facebook/bart-large-cnn"
            )

            cls._model = AutoModelForSeq2SeqLM.from_pretrained(
                "facebook/bart-large-cnn"
            ).to(cls._device)

            cls._model.eval()

        return cls._tokenizer, cls._model, cls._device


# =========================================================
# Paraphraser (T5)
# =========================================================
class Paraphraser:

    _tokenizer = None
    _model = None
    _device = None

    @classmethod
    def get(cls):

        if cls._tokenizer is None or cls._model is None:

            cls._device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

            model_name = "ramsrigouthamg/t5_paraphraser"

            cls._tokenizer = AutoTokenizer.from_pretrained(
                model_name
            )

            cls._model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name
            ).to(cls._device)

            cls._model.eval()

        return cls._tokenizer, cls._model, cls._device


# =========================================================
# Zero-shot classifier (BART MNLI)
# =========================================================
class ZeroShotClassifier:

    _pipeline = None

    @classmethod
    def get(cls):

        if cls._pipeline is None:

            device = 0 if torch.cuda.is_available() else -1

            cls._pipeline = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=device
            )

        return cls._pipeline


# =========================================================
# Sentence Transformer (MiniLM)
# =========================================================
class SentenceEmbedder:

    _model = None

    @classmethod
    def get(cls):

        if cls._model is None:

            cls._model = SentenceTransformer(
                "all-MiniLM-L6-v2"
            )

        return cls._model