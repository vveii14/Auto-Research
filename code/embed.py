"""Node text embeddings (pilot: all-MiniLM-L6-v2; upgrade to SPECTER2 later)."""
import numpy as np
from sentence_transformers import SentenceTransformer

_model = None


def model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed(texts):
    if not texts:
        return np.zeros((0, 384), dtype="float32")
    return model().encode(list(texts), normalize_embeddings=True,
                          show_progress_bar=False).astype("float32")
