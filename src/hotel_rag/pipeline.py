"""Pipeline RAG complet : encapsule ingestion + recherche + génération.

C'est le point d'entrée unique utilisé par l'API et le CLI. Il ne sait
rien de FastAPI ni du terminal : uniquement répondre à une question à
partir de la documentation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from hotel_rag.config import Settings, settings
from hotel_rag.embeddings import get_embedder
from hotel_rag.generation import build_prompt, get_generator
from hotel_rag.ingestion import load_documentation
from hotel_rag.retrieval import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RAGAnswer:
    """Réponse structurée du pipeline, avec ses sources pour la traçabilité."""

    question: str
    answer: str
    sources: pd.DataFrame

    def sources_summary(self) -> list[dict]:
        """Sources sous forme de liste de dicts, pratique pour une réponse JSON."""
        return self.sources[["source", "title", "score"]].to_dict(orient="records")


class RAGPipeline:
    """Assistant RAG : recherche les rubriques pertinentes puis génère une réponse."""

    def __init__(self, config: Settings = settings) -> None:
        self._config = config
        self._pages = load_documentation(config.data_dir)
        self._embedder = get_embedder(config.embedding_model)
        self._store = VectorStore(self._pages, self._embedder)
        self._generator = get_generator(
            config.generation_model,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.do_sample,
        )
        logger.info("Pipeline RAG prêt : %d rubriques indexées", self._store.size)

    @property
    def num_documents(self) -> int:
        return self._store.size

    def answer(self, question: str, top_k: int | None = None) -> RAGAnswer:
        """Répond à une question en s'appuyant uniquement sur les rubriques pertinentes."""
        top_k = top_k or self._config.default_top_k
        retrieved = self._store.search(question, top_k=top_k)
        context = "\n\n".join(retrieved["section"])
        prompt = build_prompt(self._config.role, context, question, self._config.consigne_prompt)
        answer_text = self._generator.ask(prompt)
        return RAGAnswer(question=question, answer=answer_text, sources=retrieved)


def load_pipeline(data_dir: Path | None = None) -> RAGPipeline:
    """Factory pratique pour un usage en script/notebook."""
    config = settings.model_copy(update={"data_dir": data_dir}) if data_dir else settings
    return RAGPipeline(config)
