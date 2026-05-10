# # app/embeddings.py
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# # Initialize embedding model
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# def create_title_embedding(title):
#     return embedding_model.encode(title)

# def create_embedding(text):
#     """
#     Create a semantic embedding for the combination of title and summary.
    
#     Args:
#         title (str)
#         summary (str)
#     Returns:
#         np.array: embedding vector
#     """
#     embedding = embedding_model.encode(text, normalize_embeddings=True)
#     return embedding

# def is_duplicate(new_embedding, existing_embeddings, threshold=0.85):
#     """
#     Check if a new article embedding is similar to existing ones.
    
#     Args:
#         new_embedding (np.array)
#         existing_embeddings (list of np.array)
#         threshold (float): similarity threshold
#     Returns:
#         bool: True if duplicate, False otherwise
#     """
#     if len(existing_embeddings) == 0:
#         return False

#     similarities = cosine_similarity(
#         [new_embedding],
#         existing_embeddings
#     )[0]

#     if max(similarities) > threshold:
#         return True

#     return False

# app/embeddings.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from models import SentenceEmbedder


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