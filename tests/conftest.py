"""Fixtures partagées entre les tests.

Principe : les tests d'ingestion et de retrieval ne doivent PAS
télécharger de vrais modèles Hugging Face (lent, réseau requis,
non déterministe pour le CI). On utilise donc un `FakeEmbedder`
qui simule l'interface d'`Embedder` avec des vecteurs déterministes
basés sur le contenu du texte.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class FakeEmbedder:
    """Simule `hotel_rag.embeddings.Embedder` sans dépendance réseau.

    Le vecteur d'un texte est dérivé d'un hash de son contenu : deux
    appels sur le même texte donnent toujours le même vecteur, et deux
    textes différents donnent (presque toujours) des vecteurs différents,
    ce qui suffit pour tester la logique de recherche.
    """

    dimension = 16

    def _vector_for(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        vector = rng.normal(size=self.dimension)
        return vector / np.linalg.norm(vector)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array([self._vector_for(t) for t in texts])

    def encode_one(self, text: str) -> np.ndarray:
        return self._vector_for(text)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def sample_pages() -> pd.DataFrame:
    """Un petit jeu de rubriques, semblable à la documentation de l'hôtel."""
    data = [
        {"source": "test.pdf", "title": "Piscine", "text": "La piscine est ouverte de 8h à 20h."},
        {"source": "test.pdf", "title": "Wifi", "text": "Le wifi est gratuit et illimité."},
        {"source": "test.pdf", "title": "Check-in", "text": "Le check-in démarre à 15h."},
    ]
    df = pd.DataFrame(data)
    df["section"] = "## " + df["title"] + "\n\n" + df["text"]
    return df
