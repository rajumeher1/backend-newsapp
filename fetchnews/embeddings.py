# fetchnews/embeddings.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fetchnews.models import SentenceEmbedder


def get_model():
    return SentenceEmbedder.get()


def create_title_embedding(title: str):
    model = get_model()
    return model.encode(title)


def create_embedding(text: str):
    """
    Create a semantic embedding for text.
    """
    model = get_model()
    return model.encode(text, normalize_embeddings=True)


def is_duplicate(new_embedding, existing_embeddings, threshold=0.85):
    """
    Check similarity with existing embeddings.
    """
    if not existing_embeddings:
        return False

    similarities = cosine_similarity(
        [new_embedding],
        existing_embeddings
    )[0]

    return max(similarities) > threshold