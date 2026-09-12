# # fetchnews/models.py

# import torch
# from transformers import (
#     AutoTokenizer,
#     AutoModelForSeq2SeqLM,
#     pipeline
# )
# from sentence_transformers import SentenceTransformer


# # =========================================================
# # Summarizer (BART CNN)
# # =========================================================
# class Summarizer:

#     _tokenizer = None
#     _model = None
#     _device = None

#     @classmethod
#     def get(cls):

#         if cls._tokenizer is None or cls._model is None:

#             cls._device = (
#                 "cuda"
#                 if torch.cuda.is_available()
#                 else "cpu"
#             )

#             cls._tokenizer = AutoTokenizer.from_pretrained(
#                 "facebook/bart-large-cnn"
#             )

#             cls._model = AutoModelForSeq2SeqLM.from_pretrained(
#                 "facebook/bart-large-cnn"
#             ).to(cls._device)

#             cls._model.eval()

#         return cls._tokenizer, cls._model, cls._device


# # =========================================================
# # Paraphraser (T5)
# # =========================================================
# class Paraphraser:

#     _tokenizer = None
#     _model = None
#     _device = None

#     @classmethod
#     def get(cls):

#         if cls._tokenizer is None or cls._model is None:

#             cls._device = (
#                 "cuda"
#                 if torch.cuda.is_available()
#                 else "cpu"
#             )

#             model_name = "google/flan-t5-base"

#             cls._tokenizer = AutoTokenizer.from_pretrained(
#                 model_name
#             )

#             cls._model = AutoModelForSeq2SeqLM.from_pretrained(
#                 model_name
#             ).to(cls._device)

#             cls._model.eval()

#         return cls._tokenizer, cls._model, cls._device


# # =========================================================
# # Zero-shot classifier (BART MNLI)
# # =========================================================
# class ZeroShotClassifier:

#     _pipeline = None

#     @classmethod
#     def get(cls):

#         if cls._pipeline is None:

#             device = 0 if torch.cuda.is_available() else -1

#             cls._pipeline = pipeline(
#                 "zero-shot-classification",
#                 model="facebook/bart-large-mnli",
#                 device=device
#             )

#         return cls._pipeline


# # =========================================================
# # Sentence Transformer (MiniLM)
# # =========================================================
# class SentenceEmbedder:

#     _model = None

#     @classmethod
#     def get(cls):

#         if cls._model is None:

#             cls._model = SentenceTransformer(
#                 "all-MiniLM-L6-v2"
#             )

#         return cls._model

# fetchnews/models.py

import logging
import threading
import time

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_MAX_LOAD_RETRIES = 3
_RETRY_BACKOFF_SECONDS = 5


def _with_retries(fn, description):
    """
    Retry a model/tokenizer download+load a few times before giving up.
    CI runners occasionally hit transient network errors when pulling
    multi-GB weights from the Hugging Face Hub; failing the whole cron
    run over a single flaky request is wasteful.
    """
    last_exc = None
    for attempt in range(1, _MAX_LOAD_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - deliberately broad, we retry then re-raise
            last_exc = exc
            logger.warning(
                "Attempt %d/%d failed while loading %s: %s",
                attempt, _MAX_LOAD_RETRIES, description, exc
            )
            if attempt < _MAX_LOAD_RETRIES:
                time.sleep(_RETRY_BACKOFF_SECONDS * attempt)
    logger.error("Giving up loading %s after %d attempts", description, _MAX_LOAD_RETRIES)
    raise last_exc


# =========================================================
# Shared base class for lazy-loaded Seq2Seq HF models
# =========================================================
class LazySeq2SeqModel:
    """
    Generic lazy-loading, thread-safe singleton wrapper for any
    AutoModelForSeq2SeqLM + AutoTokenizer pair.

    Subclasses just set `_model_name`.
    """

    _model_name = None

    _tokenizer = None
    _model = None
    _device = None
    _lock = threading.Lock()

    @classmethod
    def get(cls):

        if cls._model is None:
            with cls._lock:
                # Double-checked locking: re-check inside the lock in case
                # another thread finished loading while we were waiting.
                if cls._model is None:

                    if cls._model_name is None:
                        raise NotImplementedError(
                            f"{cls.__name__} must set `_model_name`"
                        )

                    cls._device = (
                        "cuda"
                        if torch.cuda.is_available()
                        else "cpu"
                    )

                    logger.info("Loading %s (%s) on %s", cls.__name__, cls._model_name, cls._device)

                    cls._tokenizer = _with_retries(
                        lambda: AutoTokenizer.from_pretrained(cls._model_name),
                        f"{cls.__name__} tokenizer"
                    )

                    cls._model = _with_retries(
                        lambda: AutoModelForSeq2SeqLM.from_pretrained(cls._model_name).to(cls._device),
                        f"{cls.__name__} model"
                    )

                    cls._model.eval()

                    logger.info("Finished loading %s", cls.__name__)

        return cls._tokenizer, cls._model, cls._device


# =========================================================
# Summarizer (BART CNN)
# =========================================================
class Summarizer(LazySeq2SeqModel):
    _model_name = "facebook/bart-large-cnn"


# =========================================================
# Paraphraser (T5-base, purpose-trained for paraphrasing)
# =========================================================
class Paraphraser(LazySeq2SeqModel):
    _model_name = "google/flan-t5-base"


# =========================================================
# Zero-shot classifier (BART MNLI)
# =========================================================
class ZeroShotClassifier:

    _pipeline = None
    _lock = threading.Lock()

    @classmethod
    def get(cls):

        if cls._pipeline is None:
            with cls._lock:
                if cls._pipeline is None:

                    device = 0 if torch.cuda.is_available() else -1

                    logger.info("Loading ZeroShotClassifier (facebook/bart-large-mnli) on device %s", device)

                    cls._pipeline = _with_retries(
                        lambda: pipeline(
                            "zero-shot-classification",
                            model="facebook/bart-large-mnli",
                            device=device
                        ),
                        "ZeroShotClassifier pipeline"
                    )

                    logger.info("Finished loading ZeroShotClassifier")

        return cls._pipeline


# =========================================================
# Sentence Transformer (MiniLM)
# =========================================================
class SentenceEmbedder:

    _model = None
    _device = None
    _lock = threading.Lock()

    @classmethod
    def get(cls):

        if cls._model is None:
            with cls._lock:
                if cls._model is None:

                    cls._device = (
                        "cuda"
                        if torch.cuda.is_available()
                        else "cpu"
                    )

                    logger.info("Loading SentenceEmbedder (all-MiniLM-L6-v2) on %s", cls._device)

                    cls._model = _with_retries(
                        lambda: SentenceTransformer("all-MiniLM-L6-v2", device=cls._device),
                        "SentenceEmbedder model"
                    )

                    logger.info("Finished loading SentenceEmbedder")

        return cls._model


# =========================================================
# Optional: warmup helper to avoid cold-start latency
# =========================================================
def warmup_all_models():
    """
    Call once at the start of your script/entrypoint (e.g. fetchnews/main.py)
    to eagerly load all models before processing begins. In a long-lived
    web server this avoids first-request latency; in a GitHub Actions cron
    job it front-loads all downloads/failures before you start doing work,
    so a broken model surfaces immediately instead of mid-run.
    """
    Summarizer.get()
    Paraphraser.get()
    ZeroShotClassifier.get()
    SentenceEmbedder.get()