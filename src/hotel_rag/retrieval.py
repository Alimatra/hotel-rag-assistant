"""Recherche des rubriques pertinentes par similarité d'embeddings.

Volontairement basé sur numpy plutôt qu'une vraie base vectorielle
(FAISS, Chroma...) : à l'échelle de quelques dizaines de rubriques,
un produit matriciel est suffisant, largement plus simple à auditer,
et sans dépendance ni service supplémentaire à opérer. Le jour où le
volume de documents grossit significativement (des milliers de
chunks), `VectorStore` est l'unique classe à remplacer : le reste du
pipeline (ingestion, génération, API) n'a pas à changer.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from hotel_rag.embeddings import Embedder

logger = logging.getLogger(__name__)


class VectorStore:
    """Index vectoriel en mémoire, construit à partir d'un DataFrame de rubriques."""

    def __init__(self, pages: pd.DataFrame, embedder: Embedder) -> None:
        if "section" not in pages.columns:
            raise ValueError("Le DataFrame de rubriques doit contenir une colonne 'section'")

        self._pages = pages.reset_index(drop=True)
        self._embedder = embedder
        self._embeddings = embedder.encode(self._pages["section"].tolist())
        logger.info(
            "Index vectoriel construit : %d vecteurs de dimension %d",
            self._embeddings.shape[0],
            self._embeddings.shape[1],
        )

    @property
    def size(self) -> int:
        return len(self._pages)

    def search(self, question: str, top_k: int = 2) -> pd.DataFrame:
        """Retourne les `top_k` rubriques les plus proches de la question.

        Retourne un DataFrame trié par score décroissant, avec une
        colonne `score` ajoutée (similarité cosinus, car les vecteurs
        sont normalisés).
        """
        if top_k > self.size:
            raise ValueError(f"top_k={top_k} dépasse le nombre de rubriques indexées ({self.size})")

        question_embedding = self._embedder.encode_one(question)
        similarities = self._embeddings @ question_embedding
        top_indices = np.argsort(-similarities)[:top_k]

        results = self._pages.iloc[top_indices].copy()
        results["score"] = similarities[top_indices]
        return results
