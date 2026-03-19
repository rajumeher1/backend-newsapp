# app/embeddings.py
from transformers import logging
logging.set_verbosity_error()

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def create_title_embedding(title):
    return embedding_model.encode(title)

def create_embedding(text):
    """
    Create a semantic embedding for the combination of title and summary.
    
    Args:
        title (str)
        summary (str)
    Returns:
        np.array: embedding vector
    """
    embedding = embedding_model.encode(text, normalize_embeddings=True)
    return embedding

def is_duplicate(new_embedding, existing_embeddings, threshold=0.85):
    """
    Check if a new article embedding is similar to existing ones.
    
    Args:
        new_embedding (np.array)
        existing_embeddings (list of np.array)
        threshold (float): similarity threshold
    Returns:
        bool: True if duplicate, False otherwise
    """
    if len(existing_embeddings) == 0:
        return False

    similarities = cosine_similarity(
        [new_embedding],
        existing_embeddings
    )[0]

    if max(similarities) > threshold:
        return True

    return False