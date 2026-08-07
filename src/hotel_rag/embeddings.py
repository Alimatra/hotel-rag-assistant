"""Encodage de texte en vecteurs (embeddings).

Wrapper léger autour de sentence-transformers, pour isoler le reste
du code de cette dépendance : si on change de modèle ou de librairie
d'embedding demain, seul ce fichier bouge.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """Encode des textes en vecteurs normalisés (similarité cosinus = produit scalaire)."""

    def __init__(self, model_name: str) -> None:
        logger.info("Chargement du modèle d'embedding : %s", model_name)
        self._model = SentenceTransformer(model_name)
        self.dimension: int = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode une liste de textes en une matrice (n_textes, dimension)."""
        return self._model.encode(texts, normalize_embeddings=True)

    def encode_one(self, text: str) -> np.ndarray:
        """Encode un seul texte en un vecteur (dimension,)."""
        return self.encode([text])[0]


@lru_cache(maxsize=1)
def get_embedder(model_name: str) -> Embedder:
    """Charge le modèle d'embedding une seule fois par processus (cache)."""
    return Embedder(model_name)
