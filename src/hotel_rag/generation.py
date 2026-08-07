"""Génération de réponse par LLM, à partir d'un contexte et d'une question."""

from __future__ import annotations

import logging
from functools import lru_cache

import transformers
from transformers import pipeline

logger = logging.getLogger(__name__)

transformers.logging.set_verbosity_error()  # masque les avertissements techniques


def build_prompt(role: str, context: str, question: str, consigne: str) -> str:
    """Assemble le prompt final : rôle, contexte, question, consigne anti-hallucination.

    L'ordre est important : la consigne en dernier agit comme un rappel
    juste avant que le modèle ne commence à générer, ce qui la rend
    plus efficace qu'en tête de prompt.
    """
    return f"{role}\n\n{context}\n\nQuestion d'un client : {question}\n{consigne}"


class Generator:
    """Wrapper autour d'un pipeline `text-generation` de transformers."""

    def __init__(self, model_name: str, max_new_tokens: int = 80, do_sample: bool = False) -> None:
        logger.info("Chargement du modèle de génération : %s", model_name)
        self._generator = pipeline("text-generation", model=model_name)
        self._max_new_tokens = max_new_tokens
        self._do_sample = do_sample

    def ask(self, prompt: str) -> str:
        """Envoie un prompt au LLM et retourne sa réponse texte."""
        conversation = self._generator(
            [{"role": "user", "content": prompt}],
            max_new_tokens=self._max_new_tokens,
            do_sample=self._do_sample,
        )
        return conversation[0]["generated_text"][-1]["content"]


@lru_cache(maxsize=1)
def get_generator(model_name: str, max_new_tokens: int = 80, do_sample: bool = False) -> Generator:
    """Charge le modèle de génération une seule fois par processus (cache)."""
    return Generator(model_name, max_new_tokens=max_new_tokens, do_sample=do_sample)
